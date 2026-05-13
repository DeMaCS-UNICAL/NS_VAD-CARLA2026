from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from agent.vlm import ClipExtractor


class FakeCapture:
    def __init__(self, frame_count: int, fps: float = 10.0) -> None:
        self.frames = [
            np.full((1, 1, 3), index, dtype=np.uint8)
            for index in range(frame_count)
        ]
        self.fps = fps
        self.position = 0
        self.seek_position: int | None = None

    def get(self, _property_id: int) -> float:
        return self.fps

    def set(self, _property_id: int, value: float) -> bool:
        self.position = int(value)
        self.seek_position = self.position
        return True

    def read(self) -> tuple[bool, object | None]:
        if self.position >= len(self.frames):
            return False, None

        frame = self.frames[self.position]
        self.position += 1
        return True, frame


class ClipExtractorTests(unittest.TestCase):
    def test_extract_clip_clamps_start_and_preserves_sampling(self) -> None:
        extractor = ClipExtractor(
            video_path="unused.mp4",
            max_frames_per_clip=2,
        )
        capture = FakeCapture(frame_count=5)

        clip = extractor.extract_clip(
            capture=capture,
            start_frame_id=-2,
            end_frame_id=4,
        )

        self.assertEqual(0, capture.seek_position)
        self.assertEqual(0, clip.start_frame_id)
        self.assertEqual(4, clip.end_frame_id)
        self.assertEqual(10.0, clip.source_fps)
        self.assertEqual(2, clip.sampled_frames)
        self.assertEqual([0, 3], [int(frame[0, 0, 0]) for frame in clip.frames])

    def test_open_capture_releases_and_raises_when_video_cannot_open(self) -> None:
        extractor = ClipExtractor(
            video_path="missing.mp4",
            max_frames_per_clip=2,
        )
        capture = mock.Mock()
        capture.isOpened.return_value = False

        with mock.patch(
            "agent.vlm.media.extractor.cv2.VideoCapture",
            return_value=capture,
        ):
            with self.assertRaisesRegex(RuntimeError, "Cannot reopen video: missing.mp4"):
                extractor.open_capture()

        capture.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
