from __future__ import annotations

from ..model.types import PerceptionFrame


def serialize_perception_frame(frame: PerceptionFrame) -> str:
    if not frame.facts:
        return f"{frame.frame_id} ;"
    return f"{frame.frame_id} " + "; ".join(frame.facts) + ";"
