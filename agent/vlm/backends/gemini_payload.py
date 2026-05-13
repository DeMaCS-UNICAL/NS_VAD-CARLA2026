from __future__ import annotations

import json
from typing import Any, TypedDict

from ...model.types import CandidateAnomaly


class ParsedGeminiResponse(TypedDict):
    """Validated Gemini response payload used by the classifier."""

    anomalous: bool
    reason: str


GEMINI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "anomalous": {
            "type": "boolean",
            "description": "Whether the candidate event is visually confirmed as anomalous.",
        },
        "reason": {
            "type": "string",
            "description": "A short explanation grounded in the observed frames.",
        },
    },
    "required": ["anomalous", "reason"],
}
GEMINI_GENERATION_CONFIG = {
    "response_mime_type": "application/json",
    "response_json_schema": GEMINI_RESPONSE_SCHEMA,
}


def build_gemini_prompt(
    *,
    candidate_anomaly: CandidateAnomaly,
    sampled_frames: int,
    input_mode: str,
) -> str:
    """Build the text prompt that frames Gemini's binary validation task."""
    visual_input = "video clip" if input_mode == "video" else "sampled frames"
    return (
        "You are validating a candidate anomaly. "
        "Inspect the provided visual evidence and determine whether it matches the candidate description. "
        "Respond only with JSON matching the provided schema.\n"
        f"Candidate description: {candidate_anomaly.description}\n"
        f"Frame range: {candidate_anomaly.start_frame_id} to {candidate_anomaly.end_frame_id}\n"
        f"Visual input: {visual_input}\n"
        f"Sampled frames: {sampled_frames}\n"
        "Set anomalous to true only if the visual evidence clearly supports the described anomaly. "
        "Set anomalous to false if the description is not visually supported, is ambiguous, "
        "or cannot be verified from the provided visual evidence. "
        "Keep the reason concise and grounded in the visible evidence."
    )


def parse_gemini_response(response_payload: Any) -> ParsedGeminiResponse:
    """Validate and normalize Gemini's JSON response payload."""
    prompt_feedback = getattr(response_payload, "prompt_feedback", None)
    block_reason = getattr(prompt_feedback, "block_reason", None)
    if isinstance(block_reason, str) and block_reason.strip():
        raise RuntimeError(f"Gemini response blocked: {block_reason}")

    parsed = getattr(response_payload, "parsed", None)
    if isinstance(parsed, dict):
        raw_payload = parsed
    else:
        raw_text = getattr(response_payload, "text", None)
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise RuntimeError("Gemini response does not contain a JSON payload")
        try:
            raw_payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Gemini response returned invalid JSON payload") from exc

    anomalous = raw_payload.get("anomalous")
    reason = raw_payload.get("reason")
    if not isinstance(anomalous, bool):
        raise RuntimeError("Gemini response JSON is missing boolean 'anomalous'")
    if not isinstance(reason, str) or not reason.strip():
        raise RuntimeError("Gemini response JSON is missing non-empty 'reason'")
    return {
        "anomalous": anomalous,
        "reason": reason.strip(),
    }
