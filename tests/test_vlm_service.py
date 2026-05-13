from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from agent.model.types import CandidateAnomaly
from agent.vlm import ExtractedClip, VLMConfig, VisualReasoner


class FakeLogger:
    def vlm_input(self, message: str, *, console: bool = False) -> None:
        return None

    def candidate_response(
        self,
        message: str,
        *,
        anomalous: bool,
        console: bool = False,
    ) -> None:
        return None

    def warning(self, message: str, *, console: bool = False) -> None:
        return None

    def error(self, message: str, *, console: bool = False) -> None:
        return None


class FakeCapture:
    def release(self) -> None:
        return None


def make_candidate() -> CandidateAnomaly:
    return CandidateAnomaly(
        id="car_1",
        start_frame_id=3,
        end_frame_id=5,
        description="driving wrong way",
        type="wrong_way",
    )


def make_clip() -> ExtractedClip:
    return ExtractedClip(
        frames=[np.zeros((2, 2, 3), dtype=np.uint8)],
        start_frame_id=3,
        end_frame_id=5,
        source_fps=10.0,
    )


class VisualReasonerTests(unittest.TestCase):
    def test_failed_candidates_counts_async_worker_errors(self) -> None:
        logger = FakeLogger()

        def failing_classifier(
            _candidate: CandidateAnomaly,
            _clip: ExtractedClip,
        ) -> object:
            raise RuntimeError("classifier failed")

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = VLMConfig(
                video_path=Path(tmp_dir) / "video.mp4",
                output_dir=tmp_dir,
                max_workers=1,
            )

            with mock.patch(
                "agent.vlm.service.ClipExtractor.open_capture",
                return_value=FakeCapture(),
            ), mock.patch(
                "agent.vlm.service.ClipExtractor.extract_clip",
                return_value=make_clip(),
            ):
                reasoner = VisualReasoner(
                    config=config,
                    classifier=failing_classifier,
                    logger=logger,
                )
                try:
                    self.assertTrue(reasoner.submit_candidate(make_candidate()))
                    self.assertTrue(reasoner.flush_pending(timeout_s=1.0))
                    self.assertEqual(1, reasoner.failed_candidates)
                finally:
                    reasoner.close()


if __name__ == "__main__":
    unittest.main()
