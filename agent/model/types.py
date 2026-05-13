from __future__ import annotations

import csv
from dataclasses import dataclass

from shapely.geometry import Point, Polygon


@dataclass(frozen=True, slots=True)
class Area:
    area_id: str
    area_type: str
    polygon: Polygon

    def contains(self, x: float, y: float) -> bool:
        """
        Return True if the point is inside the area or on its border.
        """
        point = Point(float(x), float(y))
        return bool(self.polygon.covers(point))


@dataclass(frozen=True, slots=True)
class CandidateAnomaly:
    id: str
    start_frame_id: int
    end_frame_id: int
    description: str
    type: str

    @classmethod
    def from_string(cls, string: str) -> CandidateAnomaly:
        """
        Build a CandidateAnomaly from a fact string.

        Supported formats:
        - candidate_anomaly(id,type,description,instant)
        - candidate_anomaly(id,type,description,start,end)
        """
        fact_string = string.strip()
        if not fact_string.startswith("candidate_anomaly("):
            raise ValueError(f"Expected candidate_anomaly fact, got: {fact_string}")

        args = _fact_args(fact_string)

        match len(args):
            case 4:
                candidate_id, candidate_type, description, instant = args
                start_frame_id, end_frame_id = _frame_range(instant)

            case 5:
                candidate_id, candidate_type, description, start, end = args
                start_frame_id, end_frame_id = _frame_range(start, end)

            case _:
                raise ValueError(
                    "Expected candidate_anomaly(id,type,description,instant) or "
                    "candidate_anomaly(id,type,description,start,end), "
                    f"got: {fact_string}"
                )

        return cls(
            id=candidate_id.strip(),
            start_frame_id=start_frame_id,
            end_frame_id=end_frame_id,
            description=description.strip(),
            type=candidate_type.strip(),
        )

    def __str__(self) -> str:
        """
        Return the fact string representation.
        """
        description = _quote_fact_arg(self.description)
        frame_range = _format_frame_range(
            self.start_frame_id,
            self.end_frame_id,
        )

        return (
            "candidate_anomaly("
            f"{self.id},{self.type},{description},{frame_range})"
        )


@dataclass(frozen=True, slots=True)
class Anomaly:
    id: str
    start_frame_id: int
    end_frame_id: int
    type: str

    @classmethod
    def from_string(cls, string: str) -> Anomaly:
        """
        Build an Anomaly from a fact string.

        Supported formats:
        - anomaly(id,type,instant)
        - anomaly(id,type,start,end)
        """
        fact_string = string.strip()
        if not fact_string.startswith("anomaly("):
            raise ValueError(f"Expected anomaly fact, got: {fact_string}")

        args = _fact_args(fact_string)

        match len(args):
            case 3:
                anomaly_id, anomaly_type, instant = args
                start_frame_id, end_frame_id = _frame_range(instant)

            case 4:
                anomaly_id, anomaly_type, start, end = args
                start_frame_id, end_frame_id = _frame_range(start, end)

            case _:
                raise ValueError(
                    "Expected anomaly(id,type,instant) or "
                    f"anomaly(id,type,start,end), got: {fact_string}"
                )

        anomaly_id = anomaly_id.strip()
        if not anomaly_id:
            raise ValueError(f"Anomaly id cannot be empty, got: {fact_string}")

        return cls(
            id=anomaly_id,
            start_frame_id=start_frame_id,
            end_frame_id=end_frame_id,
            type=anomaly_type.strip(),
        )

    def __str__(self) -> str:
        """
        Return the fact string representation.
        """
        frame_range = _format_frame_range(
            self.start_frame_id,
            self.end_frame_id,
        )

        return f"anomaly({self.id},{self.type},{frame_range})"


@dataclass(frozen=True, slots=True)
class PerceptionFrame:
    """
    Detector facts extracted from a single video frame.

    Attributes:
        frame_id: Sequential identifier of the source frame.
        facts: Object-detector facts forwarded to the reasoner.
    """

    frame_id: int
    facts: tuple[str, ...]

    def __post_init__(self) -> None:
        """
        Normalize facts to a tuple, even if a list is passed at construction time.
        """
        object.__setattr__(self, "facts", tuple(self.facts))


@dataclass(frozen=True, slots=True)
class ReasoningResult:
    """
    Output produced by the symbolic reasoning stage.

    Attributes:
        anomalies: Confirmed anomalies inferred by the reasoner.
        candidate_anomalies: Potential anomalies that need VLM validation.
    """

    anomalies: tuple[Anomaly, ...]
    candidate_anomalies: tuple[CandidateAnomaly, ...]

    def __post_init__(self) -> None:
        """
        Normalize collections to tuples, keeping the frozen dataclass truly immutable.
        """
        object.__setattr__(self, "anomalies", tuple(self.anomalies))
        object.__setattr__(
            self,
            "candidate_anomalies",
            tuple(self.candidate_anomalies),
        )


@dataclass(frozen=True, slots=True)
class VLMResult:
    """
    Verdict produced by the VLM for a candidate anomaly.

    Attributes:
        candidate: Candidate anomaly evaluated by the VLM.
        anomaly: True when the VLM confirms the candidate as an anomaly.
        reason: Natural-language justification for the verdict.
    """

    candidate: CandidateAnomaly
    anomaly: bool
    reason: str


def _fact_args(fact_string: str) -> tuple[str, ...]:
    """
    Extract and parse the arguments inside a fact string.

    Example:
    anomaly(3,wrong_way,128)
    -> ("3", "wrong_way", "128")

    csv.reader is used so quoted arguments containing commas still work.
    """
    opening_parenthesis = fact_string.find("(")
    closing_parenthesis = fact_string.rfind(")")

    if opening_parenthesis <= 0 or closing_parenthesis <= opening_parenthesis:
        raise ValueError(f"Invalid fact string: {fact_string}")

    inner = fact_string[opening_parenthesis + 1 : closing_parenthesis]

    return tuple(
        part.strip()
        for part in next(
            csv.reader(
                [inner],
                delimiter=",",
                quotechar='"',
                escapechar="\\",
                skipinitialspace=True,
            )
        )
    )


def _frame_range(start: str, end: str | None = None) -> tuple[int, int]:
    """
    Parse a frame instant or interval.

    If only start is provided, the interval is a single frame.
    The returned range is always normalized as (min, max).
    """
    start_frame_id = int(start.strip())
    end_frame_id = start_frame_id if end is None else int(end.strip())

    return (
        min(start_frame_id, end_frame_id),
        max(start_frame_id, end_frame_id),
    )


def _format_frame_range(start_frame_id: int, end_frame_id: int) -> str:
    """
    Format a frame range for a fact string.

    Single-frame intervals are serialized as one value.
    Multi-frame intervals are serialized as start,end.
    """
    if start_frame_id == end_frame_id:
        return str(start_frame_id)

    return f"{start_frame_id},{end_frame_id}"


def _quote_fact_arg(value: str) -> str:
    """
    Quote and escape a string argument for a fact.
    """
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
