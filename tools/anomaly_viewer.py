#!/usr/bin/env python3
"""Experimental viewer retained for reference only.

This utility parses the agent log to rebuild perception snapshots and DP-SR
anomaly events from [agent->dp-sr] and [dp-sr->agent] lines. It then matches
those events to video frame ids, draws annotations for the related objects,
and exports one annotated frame for each relevant frame or frame range.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.model.predicates import get_facts
from agent.model.types import Anomaly, CandidateAnomaly


SOCKET_FACTS_RE = re.compile(
    r"^\[agent->dp-sr\]\s+(\d+)\s+(.*)$"
)
DPSR_OUTPUT_RE = re.compile(r"^\[dp-sr->agent\]\s+(\{.*\})\s*$")
CANDIDATE_RESPONSE_RE = re.compile(
    r"^\[candidate-response\]\s+(?:Entity|entity|object with id)\s+"
    r"(?P<object_id>\S+)\s+"
    r"(?P<type>\S+)\s+"
    r"(?:(?:at frame\s+(?P<instant>\d+))|"
    r"(?:from frame\s+(?P<start>\d+)\s+to frame\s+(?P<end>\d+)))\s+"
    r"(?P<status>not anomalous|anomalous)\.\s+Reason:\s*(?P<reason>.*)$"
)
OBJECT_RE = re.compile(r"object\(([^,]+),([^,]+),(-?\d+),(-?\d+)\)")
NON_WORD_RE = re.compile(r"[^\w.-]+")
SORT_CHUNK_RE = re.compile(r"\d+|\D+")

ORANGE = (0, 165, 255)
RED = (0, 0, 255)
GREEN = (0, 180, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


@dataclass(frozen=True)
class ObjectSnapshot:
    object_id: str
    object_class: str
    x: int
    y: int


@dataclass(frozen=True)
class PerceptionSnapshot:
    frame_id: int
    objects: dict[str, ObjectSnapshot]


@dataclass(frozen=True)
class CandidateEvent:
    object_id: str
    type: str
    description: str
    start_frame_id: int
    end_frame_id: int
    confirmed: bool | None = None
    reason: str | None = None


@dataclass(frozen=True)
class AnomalyEvent:
    object_id: str | None
    type: str
    description: str
    start_frame_id: int
    end_frame_id: int
    answer_frame_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ParsedLog:
    frame_snapshots: list[PerceptionSnapshot]
    answer_snapshots: list[PerceptionSnapshot]
    candidate_events: list[CandidateEvent]
    anomaly_events: list[AnomalyEvent]


@dataclass(frozen=True)
class ViewerWindow:
    start_frame_id: int
    end_frame_id: int

    def contains(self, frame_id: int) -> bool:
        return self.start_frame_id <= frame_id <= self.end_frame_id


@dataclass(frozen=True)
class SnapshotSelection:
    snapshots: list[PerceptionSnapshot]
    viewer_window: ViewerWindow
    source: str = "regular"

    @property
    def used_answer_fallback(self) -> bool:
        return self.source == "answer_fallback"


# Log parsing
def parse_log(log_path: str | Path) -> ParsedLog:
    snapshots: list[PerceptionSnapshot] = []
    answer_snapshots: dict[int, PerceptionSnapshot] = {}
    candidate_records: dict[tuple[str, str, int, int], tuple[int, CandidateEvent]] = {}
    anomaly_records: dict[tuple[str | None, str, int, int, str], AnomalyEvent] = {}

    with open(log_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            snapshot = parse_socket_reasoner_line(line)
            if snapshot is not None:
                snapshots.append(snapshot)
                continue

            answer_record = extract_answer_record(line)
            if answer_record is not None:
                answer_frame_id, facts = answer_record
                answer_snapshots[answer_frame_id] = parse_snapshot_facts(
                    answer_frame_id,
                    ", ".join(facts),
                )
                for fact in facts:
                    if fact.startswith("candidate_anomaly("):
                        register_candidate_event(
                            candidate_records,
                            candidate_event_from_reasoner_atom(fact),
                            source_priority=2,
                        )
                    elif fact.startswith("anomaly("):
                        register_anomaly_event(
                            anomaly_records,
                            anomaly_event_from_reasoner_atom(
                                fact,
                                answer_frame_id,
                            ),
                        )
                continue

            candidate_response = candidate_event_from_response_line(line)
            if candidate_response is not None:
                register_candidate_event(
                    candidate_records,
                    candidate_response,
                    source_priority=1,
                )
                continue

    candidate_events = [event for _, event in candidate_records.values()]

    return ParsedLog(
        frame_snapshots=snapshots,
        answer_snapshots=list(answer_snapshots.values()),
        candidate_events=candidate_events,
        anomaly_events=list(anomaly_records.values()),
    )


def parse_socket_reasoner_line(line: str) -> PerceptionSnapshot | None:
    match = SOCKET_FACTS_RE.match(line)
    if not match:
        return None

    frame_id = int(match.group(1))
    return parse_snapshot_facts(frame_id, match.group(2))


def parse_snapshot_facts(frame_id: int, facts_text: str) -> PerceptionSnapshot:
    objects: dict[str, ObjectSnapshot] = {}

    for object_id, object_class, x, y in OBJECT_RE.findall(facts_text):
        objects[object_id] = ObjectSnapshot(
            object_id=object_id,
            object_class=object_class,
            x=int(x),
            y=int(y),
        )

    return PerceptionSnapshot(frame_id=frame_id, objects=objects)


def extract_answer_record(line: str) -> tuple[int, list[str]] | None:
    match = DPSR_OUTPUT_RE.match(line)
    if match is None:
        return None

    payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected DP-SR JSON object, got: {line}")

    timestamp = payload.get("timestamp")
    answers = payload.get("answers")
    if not isinstance(timestamp, str) or not timestamp.isdigit():
        raise ValueError(f"Expected DP-SR JSON timestamp string, got: {line}")
    if (
        not isinstance(answers, list)
        or len(answers) != 1
        or not isinstance(answers[0], str)
    ):
        raise ValueError(
            f"Expected DP-SR JSON answers list with one string, got: {line}"
        )

    return int(timestamp), get_facts(line)


def candidate_event_from_reasoner_atom(atom: str) -> CandidateEvent:
    candidate = CandidateAnomaly.from_string(atom)
    return CandidateEvent(
        object_id=candidate.id,
        type=candidate.type,
        description=candidate.description,
        start_frame_id=candidate.start_frame_id,
        end_frame_id=candidate.end_frame_id,
    )


def candidate_event_from_response_line(line: str) -> CandidateEvent | None:
    match = CANDIDATE_RESPONSE_RE.match(line)
    if match is None:
        return None

    instant = match.group("instant")
    if instant is not None:
        start_frame_id = end_frame_id = int(instant)
    else:
        start_frame_id, end_frame_id = sorted(
            (int(match.group("start")), int(match.group("end")))
        )

    return CandidateEvent(
        object_id=match.group("object_id"),
        type=match.group("type"),
        description="",
        start_frame_id=start_frame_id,
        end_frame_id=end_frame_id,
        confirmed=match.group("status") == "anomalous",
        reason=match.group("reason").strip() or None,
    )


def anomaly_event_from_reasoner_atom(atom: str, answer_frame_id: int) -> AnomalyEvent:
    anomaly = Anomaly.from_string(atom)
    return AnomalyEvent(
        object_id=anomaly.id,
        type=anomaly.type,
        description=f"id={anomaly.id}",
        start_frame_id=anomaly.start_frame_id,
        end_frame_id=anomaly.end_frame_id,
        answer_frame_ids=(answer_frame_id,),
    )


def candidate_key(event: CandidateEvent) -> tuple[str, str, int, int]:
    return (event.object_id, event.type, event.start_frame_id, event.end_frame_id)


def anomaly_key(event: AnomalyEvent) -> tuple[str | None, str, int, int, str]:
    return (
        event.object_id,
        event.type,
        event.start_frame_id,
        event.end_frame_id,
        event.description,
    )


def register_candidate_event(
    candidate_records: dict[tuple[str, str, int, int], tuple[int, CandidateEvent]],
    event: CandidateEvent,
    *,
    source_priority: int,
) -> None:
    key = candidate_key(event)
    existing = candidate_records.get(key)
    if existing is None:
        candidate_records[key] = (source_priority, event)
        return

    existing_priority, existing_event = existing
    if source_priority >= existing_priority:
        merged = merge_candidate_events(event, existing_event)
        merged_priority = source_priority
    else:
        merged = merge_candidate_events(existing_event, event)
        merged_priority = existing_priority

    candidate_records[key] = (merged_priority, merged)


def merge_candidate_events(
    primary: CandidateEvent,
    secondary: CandidateEvent,
) -> CandidateEvent:
    return replace(
        primary,
        description=primary.description or secondary.description,
        confirmed=(
            primary.confirmed
            if primary.confirmed is not None
            else secondary.confirmed
        ),
        reason=primary.reason or secondary.reason,
    )


def register_anomaly_event(
    anomaly_records: dict[tuple[str | None, str, int, int, str], AnomalyEvent],
    event: AnomalyEvent,
) -> None:
    key = anomaly_key(event)
    existing = anomaly_records.get(key)
    if existing is None:
        anomaly_records[key] = event
        return

    merged_frame_ids = tuple(
        sorted(set(existing.answer_frame_ids) | set(event.answer_frame_ids))
    )
    anomaly_records[key] = replace(existing, answer_frame_ids=merged_frame_ids)


# Snapshot selection
def select_snapshots_for_event(
    event: CandidateEvent | AnomalyEvent,
    snapshots: list[PerceptionSnapshot],
    answer_snapshots: list[PerceptionSnapshot] | None = None,
) -> SnapshotSelection:
    viewer_window = viewer_window_for_event(event)
    regular_snapshots = [
        snapshot
        for snapshot in snapshots
        if snapshot_matches_event(event, snapshot, viewer_window)
    ]

    if not isinstance(event, AnomalyEvent):
        return SnapshotSelection(
            snapshots=regular_snapshots,
            viewer_window=viewer_window,
        )

    if regular_snapshots:
        return SnapshotSelection(
            snapshots=regular_snapshots,
            viewer_window=viewer_window,
        )

    fallback_snapshots = collect_answer_snapshots_for_anomaly(event, answer_snapshots or [])
    return SnapshotSelection(
        snapshots=fallback_snapshots,
        viewer_window=viewer_window,
        source="answer_fallback" if fallback_snapshots else "regular",
    )


def snapshot_matches_event(
    event: CandidateEvent | AnomalyEvent,
    snapshot: PerceptionSnapshot,
    viewer_window: ViewerWindow,
) -> bool:
    return (
        viewer_window.contains(snapshot.frame_id)
        and snapshot_contains_event_target(snapshot, event)
    )


def snapshot_contains_event_target(
    snapshot: PerceptionSnapshot,
    event: CandidateEvent | AnomalyEvent,
) -> bool:
    target_object_id = event_target_object_id(event)
    return target_object_id is None or target_object_id in snapshot.objects


def collect_answer_snapshots_for_anomaly(
    event: AnomalyEvent,
    answer_snapshots: list[PerceptionSnapshot],
) -> list[PerceptionSnapshot]:
    trigger_frame_id = anomaly_answer_trigger_frame_id(event)
    if trigger_frame_id is None:
        return []

    snapshots_by_frame_id = {
        snapshot.frame_id: snapshot
        for snapshot in answer_snapshots
    }
    snapshot = snapshots_by_frame_id.get(trigger_frame_id)
    if snapshot is None:
        return []

    if event.object_id is not None and event.object_id not in snapshot.objects:
        return []

    return [snapshot]


# Frame annotation
def draw_event_frame(
    frame,
    snapshot: PerceptionSnapshot,
    event: CandidateEvent | AnomalyEvent,
    candidate_events: list[CandidateEvent],
    anomaly_events: list[AnomalyEvent],
    *,
    selection: SnapshotSelection,
    show_annotations: bool = True,
):
    if not show_annotations:
        return frame

    annotations = related_events_for_snapshot(
        event,
        snapshot,
        candidate_events,
        anomaly_events,
    )
    target_object = snapshot.objects.get(event_target_object_id(event) or "")

    if target_object is not None:
        draw_object_annotations(frame, target_object, annotations)

    header_lines = build_event_header_lines(event, snapshot)
    header_height = draw_text_box(frame, header_lines)
    warning_lines = build_event_warning_lines(
        event,
        selection,
    )
    if warning_lines:
        draw_text_box(
            frame,
            ["WARNING", *warning_lines],
            origin=(16, 16 + header_height + 12),
            border_color=RED,
            text_color=ORANGE,
        )
    return frame


def related_events_for_snapshot(
    event: CandidateEvent | AnomalyEvent,
    snapshot: PerceptionSnapshot,
    candidate_events: list[CandidateEvent],
    anomaly_events: list[AnomalyEvent],
) -> list[CandidateEvent | AnomalyEvent]:
    target_object_id = event_target_object_id(event)
    if target_object_id is None:
        return [event]

    related: list[CandidateEvent | AnomalyEvent] = [event]
    seen = {event_identity(event)}

    for candidate in candidate_events:
        if candidate.object_id != target_object_id:
            continue
        if not candidate.start_frame_id <= snapshot.frame_id <= candidate.end_frame_id:
            continue
        candidate_id = event_identity(candidate)
        if candidate_id not in seen:
            related.append(candidate)
            seen.add(candidate_id)

    for anomaly in anomaly_events:
        if anomaly.object_id != target_object_id:
            continue
        if not anomaly.start_frame_id <= snapshot.frame_id <= anomaly.end_frame_id:
            continue
        anomaly_id = event_identity(anomaly)
        if anomaly_id not in seen:
            related.append(anomaly)
            seen.add(anomaly_id)

    return related


def draw_object_annotations(
    frame,
    obj: ObjectSnapshot,
    annotations: list[CandidateEvent | AnomalyEvent],
) -> None:
    x = obj.x
    y = obj.y

    candidate_color = candidate_annotations_color(annotations)
    has_anomaly = any(
        isinstance(annotation, AnomalyEvent) and annotation.object_id is not None
        for annotation in annotations
    )

    if candidate_color is not None:
        cv2.circle(frame, (x, y), 7, candidate_color, -1)
    if has_anomaly:
        cv2.circle(frame, (x, y), 12, RED, 2)

    label_y = y - 12
    for text, color in build_object_label_lines(annotations):
        cv2.putText(
            frame,
            text,
            (x + 12, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        label_y += 20

    cv2.putText(
        frame,
        f"id={obj.object_id} {obj.object_class}",
        (x + 12, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        WHITE,
        1,
        cv2.LINE_AA,
    )


def build_object_label_lines(
    annotations: list[CandidateEvent | AnomalyEvent],
) -> list[tuple[str, tuple[int, int, int]]]:
    lines: list[tuple[str, tuple[int, int, int]]] = []

    for annotation in annotations:
        if isinstance(annotation, CandidateEvent):
            lines.append(
                (
                    f"candidate: {annotation.type} "
                    f"({format_candidate_confirmation(annotation)})",
                    candidate_confirmation_color(annotation),
                )
            )
        else:
            lines.append((f"anomaly: {annotation.type}", RED))

    return lines


def build_event_header_lines(
    event: CandidateEvent | AnomalyEvent,
    snapshot: PerceptionSnapshot,
) -> list[str]:
    lines = [f"frame_id: {snapshot.frame_id}"]

    if isinstance(event, CandidateEvent):
        lines.append(f"frame range in candidate: {format_event_frame_range(event)}")
        lines.append(f"candidate: {event.type} object={event.object_id}")
        lines.append(f"confirmed: {format_candidate_confirmation(event)}")
        lines.append(f"reason: {format_candidate_reason(event)}")
        if event.description:
            lines.append(f"description: {event.description}")
        return lines

    target = event.object_id if event.object_id is not None else "global"
    lines.append(f"frame range in anomaly: {format_event_frame_range(event)}")
    lines.extend(build_anomaly_answer_lines(event))
    lines.append(f"anomaly: {event.type} object={target}")
    if event.description and event.description != f"id={event.object_id}":
        lines.append(f"description: {event.description}")
    return lines


def build_anomaly_answer_lines(event: AnomalyEvent) -> list[str]:
    trigger_frame_id = anomaly_answer_trigger_frame_id(event)
    if trigger_frame_id is None:
        return []

    return [f"answer trigger frame_id: {trigger_frame_id}"]


def format_candidate_confirmation(event: CandidateEvent) -> str:
    if event.confirmed is None:
        return "unknown"
    return "yes" if event.confirmed else "no"


def candidate_annotations_color(
    annotations: list[CandidateEvent | AnomalyEvent],
) -> tuple[int, int, int] | None:
    candidates = [
        annotation
        for annotation in annotations
        if isinstance(annotation, CandidateEvent)
    ]
    if not candidates:
        return None
    if any(candidate.confirmed is True for candidate in candidates):
        return RED
    if all(candidate.confirmed is False for candidate in candidates):
        return GREEN
    return ORANGE


def candidate_confirmation_color(event: CandidateEvent) -> tuple[int, int, int]:
    if event.confirmed is True:
        return RED
    if event.confirmed is False:
        return GREEN
    return ORANGE


def format_candidate_reason(event: CandidateEvent) -> str:
    if event.reason:
        return event.reason
    if event.confirmed is None:
        return "no candidate-response found in log"
    return "not provided"


def build_event_warning_lines(
    event: CandidateEvent | AnomalyEvent,
    selection: SnapshotSelection,
) -> list[str]:
    if not isinstance(event, AnomalyEvent):
        return []

    lines: list[str] = []
    if event.object_id is None:
        lines.append(
            "The anomaly description did not expose an object id, so this event is treated as global."
        )
    if selection.used_answer_fallback:
        lines.append(
            "This frame comes from a reasoner ANSWER snapshot fallback because no regular frame snapshot at the answer trigger frame_id contained the anomaly object."
        )

    return lines


def draw_text_box(
    frame,
    lines: list[str],
    *,
    origin: tuple[int, int] = (16, 16),
    width_chars: int = 56,
    border_color: tuple[int, int, int] = WHITE,
    text_color: tuple[int, int, int] = WHITE,
    fill_color: tuple[int, int, int] = BLACK,
) -> int:
    wrapped_lines: list[str] = []
    for line in lines:
        wrapped = textwrap.wrap(line, width=width_chars) or [""]
        wrapped_lines.extend(wrapped)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    line_height = 18
    padding = 10

    text_sizes = [
        cv2.getTextSize(text, font, font_scale, thickness)[0]
        for text in wrapped_lines
    ]
    box_width = max((size[0] for size in text_sizes), default=0) + padding * 2
    box_height = len(wrapped_lines) * line_height + padding * 2

    x, y = origin
    cv2.rectangle(
        frame,
        (x, y),
        (x + box_width, y + box_height),
        fill_color,
        -1,
    )
    cv2.rectangle(
        frame,
        (x, y),
        (x + box_width, y + box_height),
        border_color,
        1,
    )

    baseline_y = y + padding + 12
    for text in wrapped_lines:
        cv2.putText(
            frame,
            text,
            (x + padding, baseline_y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
        baseline_y += line_height

    return box_height


def event_identity(event: CandidateEvent | AnomalyEvent) -> tuple[str, str | None, str, int, int]:
    return (
        event_kind(event),
        event_target_object_id(event),
        event.type,
        event.start_frame_id,
        event.end_frame_id,
    )


def event_kind(event: CandidateEvent | AnomalyEvent) -> str:
    return "candidate" if isinstance(event, CandidateEvent) else "anomaly"


def event_target_object_id(event: CandidateEvent | AnomalyEvent) -> str | None:
    return event.object_id


def natural_sort_key(value: str) -> tuple[tuple[int, int | str], ...]:
    parts: list[tuple[int, int | str]] = []
    for chunk in SORT_CHUNK_RE.findall(value):
        if chunk.isdigit():
            parts.append((0, int(chunk)))
        else:
            parts.append((1, chunk.lower()))
    return tuple(parts)


def event_sort_key_for_show(
    event: CandidateEvent | AnomalyEvent,
) -> tuple[bool, tuple[tuple[int, int | str], ...], int, int, int, str]:
    object_id = event_target_object_id(event) or ""
    kind_rank = 0 if isinstance(event, CandidateEvent) else 1
    return (
        event_target_object_id(event) is None,
        natural_sort_key(object_id),
        event.start_frame_id,
        event.end_frame_id,
        kind_rank,
        event.type.lower(),
    )


def anomaly_answer_trigger_frame_id(event: AnomalyEvent) -> int | None:
    if not event.answer_frame_ids:
        return None
    return min(event.answer_frame_ids)


def format_event_frame_range(event: CandidateEvent | AnomalyEvent) -> str:
    if event.start_frame_id == event.end_frame_id:
        return str(event.start_frame_id)
    return f"{event.start_frame_id}-{event.end_frame_id}"


def viewer_window_for_event(event: CandidateEvent | AnomalyEvent) -> ViewerWindow:
    return ViewerWindow(event.start_frame_id, event.end_frame_id)


def event_output_dir(base_dir: Path, event: CandidateEvent | AnomalyEvent) -> Path:
    object_id = event_target_object_id(event) or "global"
    prefix = (
        f"{event_kind(event)}_{slugify(event.type)}_object-{slugify(object_id)}"
    )
    if event.start_frame_id == event.end_frame_id:
        return base_dir / prefix
    return base_dir / f"{prefix}_{event.start_frame_id}-{event.end_frame_id}"


def slugify(value: str) -> str:
    cleaned = NON_WORD_RE.sub("_", value).strip("_")
    return cleaned or "unknown"


def describe_event(event: CandidateEvent | AnomalyEvent) -> str:
    target = event_target_object_id(event) or "global"
    return (
        f"{event_kind(event)} type={event.type} "
        f"object={target} frame_range={event.start_frame_id}-{event.end_frame_id}"
    )


# Rendering and CLI
def render_event(
    cap: cv2.VideoCapture,
    event: CandidateEvent | AnomalyEvent,
    parsed_log: ParsedLog,
    output_dir: Path,
    *,
    show: bool = False,
    show_annotations: bool = True,
) -> bool:
    selection = select_snapshots_for_event(
        event,
        parsed_log.frame_snapshots,
        parsed_log.answer_snapshots,
    )
    snapshots = sorted(selection.snapshots, key=lambda snapshot: snapshot.frame_id)
    if not snapshots:
        print(missing_frames_message(event))
        return False

    event_dir = event_output_dir(output_dir, event)
    event_dir.mkdir(parents=True, exist_ok=True)

    for snapshot in snapshots:
        cap.set(cv2.CAP_PROP_POS_FRAMES, snapshot.frame_id)
        ok, frame = cap.read()
        if not ok:
            print(f"Frame not found for frame_id {snapshot.frame_id}")
            continue

        annotated = draw_event_frame(
            frame,
            snapshot,
            event,
            parsed_log.candidate_events,
            parsed_log.anomaly_events,
            selection=selection,
            show_annotations=show_annotations,
        )
        out_path = event_dir / f"frame_{snapshot.frame_id}.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"Saved: {out_path}")

        if show:
            cv2.imshow("Annotated frame", annotated)
            key = cv2.waitKey(0)
            if key == 27:
                return True

    return False


def missing_frames_message(event: CandidateEvent | AnomalyEvent) -> str:
    base = f"No matching frames for {describe_event(event)}"
    if not isinstance(event, AnomalyEvent):
        return base

    trigger_frame_id = anomaly_answer_trigger_frame_id(event)
    trigger_suffix = (
        f" at answer trigger frame_id {trigger_frame_id}"
        if trigger_frame_id is not None
        else ""
    )

    if event.object_id is None:
        return (
            f"{base}. No frame or reasoner ANSWER snapshot was available{trigger_suffix}."
        )

    return (
        f"{base}. Object {event.object_id} was not found in regular frame "
        f"snapshots or reasoner ANSWER snapshots{trigger_suffix}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Utility to visualize frame-based anomaly events on a video using "
            "the new agent.log format. The tool reads [agent->dp-sr] frame "
            "snapshots and [dp-sr->agent] JSON events, then "
            "exports one annotated frame for each frame_id in an event range."
        )
    )
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--log", required=True, help="Path to the agent log file")
    parser.add_argument(
        "--out",
        default="./outputs/annotated_frames",
        help="Output directory for per-event annotated frames",
    )
    parser.add_argument("--show", action="store_true", help="Display annotated frames")
    parser.add_argument(
        "--hide-annotations",
        action="store_true",
        help="Export and display raw frames without drawing annotations",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    log_path = Path(args.log)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed_log = parse_log(log_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        stop_requested = False
        events: list[CandidateEvent | AnomalyEvent] = [
            *parsed_log.candidate_events,
            *parsed_log.anomaly_events,
        ]
        if not events:
            print("No anomaly or candidate events found in the log")
            return

        if args.show:
            events.sort(key=event_sort_key_for_show)

        for event in events:
            stop_requested = render_event(
                cap,
                event,
                parsed_log,
                output_dir,
                show=args.show,
                show_annotations=not args.hide_annotations,
            )
            if stop_requested:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
