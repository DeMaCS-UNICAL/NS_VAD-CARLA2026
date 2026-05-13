from __future__ import annotations

from typing import Protocol

from ..model.types import CandidateAnomaly, VLMResult
from .models import ExtractedClip


class VisualLanguageClassifier(Protocol):
    """Callable interface implemented by VLM validation backends."""

    def __call__(
        self,
        candidate_anomaly: CandidateAnomaly,
        clip: ExtractedClip,
    ) -> VLMResult:
        """Classify whether the clip visually confirms a candidate anomaly."""
        ...


ClipClassifier = VisualLanguageClassifier
