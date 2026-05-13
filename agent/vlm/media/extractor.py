from __future__ import annotations

import math

import cv2

from ..models import ExtractedClip, FrameImage


class ClipExtractor:
    """Extract sampled frame clips from a video source using OpenCV."""

    def __init__(
        self,
        *,
        video_path: str,
        max_frames_per_clip: int,
    ) -> None:
        self._video_path = video_path
        self._max_frames_per_clip = max_frames_per_clip

    def open_capture(self) -> cv2.VideoCapture:
        """Open a fresh capture for the configured video or raise a clear error."""
        capture = cv2.VideoCapture(self._video_path)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot reopen video: {self._video_path}")
        return capture

    def extract_clip(
        self,
        *,
        capture: cv2.VideoCapture,
        start_frame_id: int,
        end_frame_id: int,
    ) -> ExtractedClip:
        """Sample frames from the requested inclusive frame range."""
        frames: list[FrameImage] = []
        start_frame_id = max(0, start_frame_id)
        end_frame_id = max(start_frame_id, end_frame_id)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_id)
        target_count = max(1, end_frame_id - start_frame_id + 1)
        stride = max(1, math.ceil(target_count / self._max_frames_per_clip))

        frame_index = start_frame_id
        while frame_index <= end_frame_id:
            ok, frame = capture.read()
            if not ok:
                break
            if (frame_index - start_frame_id) % stride == 0:
                frames.append(frame)
            frame_index += 1

        return ExtractedClip(
            frames=frames,
            start_frame_id=start_frame_id,
            end_frame_id=end_frame_id,
            source_fps=self._source_fps(capture),
        )

    def _source_fps(self, capture: cv2.VideoCapture) -> float:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            raise RuntimeError(f"OpenCV cannot read FPS from video: {self._video_path}")
        return float(fps)
