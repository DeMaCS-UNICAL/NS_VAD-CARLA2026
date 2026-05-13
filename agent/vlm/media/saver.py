from __future__ import annotations

import re
from pathlib import Path

import cv2

from ...logging.events import EventLogger
from ...model.types import CandidateAnomaly
from ..models import ExtractedClip
from .writer import write_clip_video


NON_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class VlmInputSaver:
    """Persist the visual evidence sent to the VLM for inspection/debugging."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        vlm_input_mode: str,
        enabled: bool,
        logger: EventLogger | None = None,
    ) -> None:
        self._save_input_dir = Path(output_dir) / "vlm_input" if enabled else None
        self._vlm_input_mode = vlm_input_mode
        self._logger = logger

    def save(
        self,
        candidate: CandidateAnomaly,
        clip: ExtractedClip,
    ) -> None:
        """Save a candidate clip as frames or MP4 when persistence is enabled."""
        if self._save_input_dir is None:
            return

        try:
            event_dir = self._save_input_dir / self._candidate_dir_name(candidate)
            event_dir.mkdir(parents=True, exist_ok=True)
            if self._vlm_input_mode == "frames":
                saved_path = self._save_frames(event_dir, clip)
            else:
                saved_path = event_dir / "input.mp4"
                write_clip_video(clip, saved_path, empty_clip_ok=True)
            self._log_vlm_input(f"saved_input={saved_path}")
        except Exception as exc:
            self._log_warning(f"source=vlm save_input_error={exc}", console=True)

    def _save_frames(self, event_dir: Path, clip: ExtractedClip) -> Path:
        for index, frame in enumerate(clip.frames):
            frame_path = event_dir / f"frame_{index:04d}.jpg"
            written = cv2.imwrite(
                str(frame_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 85],
            )
            if not written:
                raise RuntimeError(f"failed to write frame for {frame_path}")
        return event_dir

    def _candidate_dir_name(self, candidate: CandidateAnomaly) -> str:
        return (
            f"{self._safe_filename(candidate.type)}_object-"
            f"{self._safe_filename(candidate.id)}_"
            f"{candidate.start_frame_id}-{candidate.end_frame_id}"
        )

    @staticmethod
    def _safe_filename(value: str) -> str:
        cleaned = NON_FILENAME_RE.sub("_", value).strip("_")
        return cleaned or "unknown"

    def _log_vlm_input(
        self,
        message: str,
        *,
        console: bool = False,
    ) -> None:
        if self._logger is not None:
            self._logger.vlm_input(message, console=console)

    def _log_warning(
        self,
        message: str,
        *,
        console: bool = False,
    ) -> None:
        if self._logger is not None:
            self._logger.warning(message, console=console)
