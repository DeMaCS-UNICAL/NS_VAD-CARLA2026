from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from agent.model.types import CandidateAnomaly
from agent.vlm import (
    DisabledVLMClassifier,
    ExtractedClip,
    build_visual_language_classifier,
)


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
        frames=[np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)],
        start_frame_id=3,
        end_frame_id=5,
        source_fps=10.0,
    )


class DisabledVLMClassifierTests(unittest.TestCase):
    def test_disabled_classifier_preserves_fallback_response(self) -> None:
        classifier = DisabledVLMClassifier(input_mode="video")

        result = classifier(make_candidate(), make_clip())

        self.assertFalse(result.anomaly)
        self.assertEqual(make_candidate(), result.candidate)
        self.assertEqual(
            "VLM backend not configured; vlm_input_mode=video sampled_frames=2",
            result.reason,
        )


class ClassifierFactoryTests(unittest.TestCase):
    def test_missing_gemini_key_uses_disabled_classifier(self) -> None:
        classifier = build_visual_language_classifier(
            gemini_api_key=" ",
            gemini_model="model",
            vlm_input_mode="frames",
        )

        self.assertIsInstance(classifier, DisabledVLMClassifier)
        result = classifier(make_candidate(), make_clip())
        self.assertEqual(
            "VLM backend not configured; vlm_input_mode=frames sampled_frames=2",
            result.reason,
        )

    def test_gemini_key_builds_classifier(self) -> None:
        gemini_classifier = mock.Mock()
        with mock.patch(
            "agent.vlm.factory.GeminiVLMClassifier",
            return_value=gemini_classifier,
        ) as classifier_class:
            classifier = build_visual_language_classifier(
                gemini_api_key=" key ",
                gemini_model="gemini-model",
                vlm_input_mode="video",
            )

            self.assertIs(gemini_classifier, classifier)

        classifier_class.assert_called_once_with(
            api_key=" key ",
            model="gemini-model",
            input_mode="video",
        )


if __name__ == "__main__":
    unittest.main()
