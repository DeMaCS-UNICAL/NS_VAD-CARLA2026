from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from agent.model.types import CandidateAnomaly
from agent.vlm import ExtractedClip, VlmInputSaver


class FakeLogger:
    def vlm_input(self, message: str, *, console: bool = False) -> None:
        return None

    def warning(self, message: str, *, console: bool = False) -> None:
        return None


def make_candidate() -> CandidateAnomaly:
    return CandidateAnomaly(
        id="car/1",
        start_frame_id=3,
        end_frame_id=5,
        description="driving wrong way",
        type="wrong way!",
    )


def make_clip() -> ExtractedClip:
    frames = [
        np.full((2, 2, 3), index, dtype=np.uint8)
        for index in range(2)
    ]
    return ExtractedClip(
        frames=frames,
        start_frame_id=3,
        end_frame_id=5,
        source_fps=10.0,
    )


class VlmInputSaverTests(unittest.TestCase):
    def test_disabled_saver_does_not_create_output(self) -> None:
        logger = FakeLogger()
        with tempfile.TemporaryDirectory() as tmp_dir:
            saver = VlmInputSaver(
                output_dir=tmp_dir,
                vlm_input_mode="frames",
                enabled=False,
                logger=logger,
            )

            saver.save(make_candidate(), make_clip())

            self.assertFalse((Path(tmp_dir) / "vlm_input").exists())

    def test_save_frames_uses_existing_directory_format(self) -> None:
        logger = FakeLogger()
        with tempfile.TemporaryDirectory() as tmp_dir:
            saver = VlmInputSaver(
                output_dir=tmp_dir,
                vlm_input_mode="frames",
                enabled=True,
                logger=logger,
            )
            event_dir = (
                Path(tmp_dir) / "vlm_input" / "wrong_way_object-car_1_3-5"
            )

            saver.save(make_candidate(), make_clip())

            self.assertTrue((event_dir / "frame_0000.jpg").exists())
            self.assertTrue((event_dir / "frame_0001.jpg").exists())


if __name__ == "__main__":
    unittest.main()
