from __future__ import annotations

from pathlib import Path

import cv2

from ..models import ExtractedClip


def write_clip_video(
    clip: ExtractedClip,
    output_path: str | Path,
    *,
    empty_clip_ok: bool = False,
    open_error_message: str | None = None,
) -> None:
    """Encode an extracted clip to an MP4 file using its playback frame rate."""
    if not clip.frames:
        if empty_clip_ok:
            return
        raise RuntimeError("Cannot write video clip without frames")

    first_frame = clip.frames[0]
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        clip.playback_fps,
        (width, height),
    )
    if not writer.isOpened():
        message = open_error_message or f"failed to open video writer for {output_path}"
        raise RuntimeError(message)

    try:
        for frame in clip.frames:
            writer.write(frame)
    finally:
        writer.release()
