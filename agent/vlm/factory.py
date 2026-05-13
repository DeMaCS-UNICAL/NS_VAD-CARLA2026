from __future__ import annotations

from .backends.disabled import DisabledVLMClassifier
from .backends.gemini import GeminiVLMClassifier
from .protocols import VisualLanguageClassifier


def build_visual_language_classifier(
    *,
    gemini_api_key: str | None,
    gemini_model: str,
    vlm_input_mode: str,
) -> VisualLanguageClassifier:
    """Build the configured classifier, falling back to a disabled backend."""
    if gemini_api_key is None or not gemini_api_key.strip():
        return DisabledVLMClassifier(input_mode=vlm_input_mode)

    return GeminiVLMClassifier(
        api_key=gemini_api_key,
        model=gemini_model,
        input_mode=vlm_input_mode,
    )
