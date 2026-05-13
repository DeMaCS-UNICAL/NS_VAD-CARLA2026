from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "Agent": (".runtime.application", "Agent"),
    "Anomaly": (".model.types", "Anomaly"),
    "Area": (".model.types", "Area"),
    "CandidateAnomaly": (".model.types", "CandidateAnomaly"),
    "DEFAULT_GEMINI_MODEL": (".vlm.backends.gemini", "DEFAULT_GEMINI_MODEL"),
    "DpSrReasoner": (".reasoning.reasoner", "DpSrReasoner"),
    "EventLogger": (".logging.events", "EventLogger"),
    "GeminiVLMClassifier": (".vlm.backends.gemini", "GeminiVLMClassifier"),
    "Perception": (".vision.perception", "Perception"),
    "PerceptionFrame": (".model.types", "PerceptionFrame"),
    "ReasoningResult": (".model.types", "ReasoningResult"),
    "VisualReasoner": (".vlm.service", "VisualReasoner"),
    "VLMResult": (".model.types", "VLMResult"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
