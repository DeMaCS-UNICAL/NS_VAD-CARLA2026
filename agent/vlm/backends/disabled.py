from __future__ import annotations

from ...model.types import CandidateAnomaly, VLMResult
from ..models import ExtractedClip
from ..protocols import VisualLanguageClassifier


class DisabledVLMClassifier(VisualLanguageClassifier):
    """Classifier used when no external VLM backend is configured."""

    def __init__(self, input_mode: str) -> None:
        self._input_mode = input_mode

    def __call__(
        self,
        candidate_anomaly: CandidateAnomaly,
        clip: ExtractedClip,
    ) -> VLMResult:
        """Return a deterministic non-anomalous result with diagnostic context."""
        return VLMResult(
            candidate=candidate_anomaly,
            anomaly=False,
            reason=(
                "VLM backend not configured; "
                f"vlm_input_mode={self._input_mode} "
                f"sampled_frames={clip.sampled_frames}"
            ),
        )
