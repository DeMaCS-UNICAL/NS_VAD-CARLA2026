from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FrameImage = NDArray[np.uint8]


@dataclass(frozen=True)
class ExtractedClip:
    """Sampled visual evidence associated with a candidate anomaly."""

    frames: list[FrameImage]
    start_frame_id: int
    end_frame_id: int
    source_fps: float

    @property
    def sampled_frames(self) -> int:
        """Number of frames retained after clip sampling."""
        return len(self.frames)

    @property
    def playback_fps(self) -> float:
        """FPS that preserves the clip duration when sampled frames are encoded."""
        source_frame_count = max(1, self.end_frame_id - self.start_frame_id + 1)
        return max(1.0, self.sampled_frames / (source_frame_count / self.source_fps))
