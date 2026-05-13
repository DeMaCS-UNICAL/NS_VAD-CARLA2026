from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from shapely.geometry import Polygon
from ultralytics import YOLO

from ..common.text import sanitize_atom
from ..model.types import Area, PerceptionFrame

if TYPE_CHECKING:
    from ..logging.events import EventLogger


class Perception:
    def __init__(
        self,
        video_path: str,
        model_path: str,
        yolo_device: str,
        areas: list[Area],
        logger: EventLogger | None = None,
    ) -> None:
        self._areas = list(areas)
        self._area_facts = [f"area({area.area_id},{area.area_type})" for area in self._areas]
        self._logger = logger
        self._capture = cv2.VideoCapture(video_path)
        if not self._capture.isOpened():
            raise RuntimeError(f"OpenCV cannot open video: {video_path}")

        fps = self._capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            self._capture.release()
            raise RuntimeError(f"OpenCV cannot read FPS from video: {video_path}")

        self._model = YOLO(model_path)
        self._class_names = getattr(self._model, "names", {})
        self._fps = float(fps)
        self._yolo_device = yolo_device.strip()
        if not self._yolo_device:
            self._capture.release()
            raise ValueError("YOLO device cannot be empty")
        self._next_frame_index = 0
        self._ended = False

    @staticmethod
    def load_areas_from_json(path: str) -> list[Area]:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        entries = payload.get("areas") if isinstance(payload, dict) else payload
        if not isinstance(entries, list) or not entries:
            raise ValueError("Areas file must contain a non-empty 'areas' list")

        areas: list[Area] = []
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid area entry at index {index}")

            area_id = sanitize_atom(entry.get("id", f"area_{index}"))
            area_type = sanitize_atom(entry.get("type", "generic"))
            raw_polygon = entry.get("polygon")

            if not isinstance(raw_polygon, list) or len(raw_polygon) < 3:
                raise ValueError(
                    f"Area '{area_id}' must define a polygon with at least 3 points"
                )

            points: list[tuple[float, float]] = []
            for point_index, raw_point in enumerate(raw_polygon):
                if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
                    raise ValueError(
                        f"Area '{area_id}' has invalid point at index {point_index}"
                    )
                points.append((float(raw_point[0]), float(raw_point[1])))

            areas.append(Area(area_id=area_id, area_type=area_type, polygon=Polygon(points)))

        return areas

    def has_ended(self) -> bool:
        return self._ended

    def next_frame(self) -> PerceptionFrame:
        if self._ended:
            raise StopIteration("Perception stream has ended")

        ok, frame = self._capture.read()
        if not ok:
            self._ended = True
            raise StopIteration("No more frames")

        frame_id = self._next_frame_index
        self._next_frame_index += 1

        result = self._model.track(
            frame,
            device=self._yolo_device,
            persist=True,
            verbose=False,
        )[0]
        facts = list(self._area_facts)
        boxes = result.boxes
        detected_by_class: dict[str, list[int]] = defaultdict(list)

        if boxes is not None and len(boxes) > 0:
            xyxy_list = boxes.xyxy.cpu().tolist()
            cls_list = boxes.cls.cpu().tolist()
            track_ids = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(xyxy_list)

            for local_index, (xyxy, cls_id, track_id) in enumerate(
                zip(xyxy_list, cls_list, track_ids),
                start=1,
            ):
                x1, y1, x2, y2 = [float(value) for value in xyxy]

                centroid_x = int(round((x1 + x2) / 2.0))
                centroid_y = int(round((y1 + y2) / 2.0))

                object_id = int(track_id) if track_id is not None else local_index
                class_name = sanitize_atom(self._resolve_class_name(int(cls_id)))
                detected_by_class[class_name].append(object_id)

                facts.append(f"object({object_id},{class_name},{centroid_x},{centroid_y})")

                for area in self._areas:
                    if area.contains(centroid_x, centroid_y):
                        facts.append(f"in_area({area.area_id},{object_id})")

        self._log_detected_objects(frame_id, detected_by_class)

        return PerceptionFrame(
            frame_id=frame_id,
            facts=tuple(facts),
        )

    def close(self) -> None:
        self._capture.release()

    @property
    def fps(self) -> float:
        return self._fps

    def _resolve_class_name(self, class_id: int) -> str:
        if isinstance(self._class_names, dict):
            return str(self._class_names.get(class_id, class_id))
        if isinstance(self._class_names, list) and 0 <= class_id < len(self._class_names):
            return str(self._class_names[class_id])
        return str(class_id)

    def _log_detected_objects(
        self,
        frame_id: int,
        detected_by_class: dict[str, list[int]],
    ) -> None:
        if self._logger is None:
            return

        total_objects = sum(len(object_ids) for object_ids in detected_by_class.values())
        if total_objects == 0:
            return

        type_summaries = []
        for class_name, object_ids in sorted(detected_by_class.items()):
            formatted_ids = ", ".join(str(object_id) for object_id in object_ids)
            type_summaries.append(
                f"{class_name} ({len(object_ids)}; ids: {formatted_ids})"
            )

        self._logger.perception(
            (
                f"Found {total_objects} objects, types: "
                f"{', '.join(type_summaries)} frame_id={frame_id}"
            ),
            console=True,
            persist=False,
        )
