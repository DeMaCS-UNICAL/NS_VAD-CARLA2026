from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_VLM_INPUT_MODES = frozenset({"frames", "video"})


@dataclass(frozen=True)
class VLMConfig:
    """Runtime settings for VLM clip extraction, persistence, and workers."""

    video_path: str | Path
    vlm_input_mode: str = "video"
    save_vlm_input: bool = False
    output_dir: str | Path = Path("outputs")
    max_workers: int = 1
    max_frames_per_clip: int = 32
    worker_join_timeout_s: float = 1.0

    def __post_init__(self) -> None:
        """Normalize filesystem paths and reject unsupported runtime values."""
        object.__setattr__(self, "video_path", Path(self.video_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))

        if self.vlm_input_mode not in SUPPORTED_VLM_INPUT_MODES:
            raise ValueError(f"Unsupported VLM input mode: {self.vlm_input_mode}")
        if self.max_workers < 1:
            raise ValueError("VLM workers must be at least 1")
        if self.max_frames_per_clip < 1:
            raise ValueError("max_frames_per_clip must be at least 1")
