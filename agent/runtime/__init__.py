from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "Agent": (".application", "Agent"),
    "build_parser": (".cli", "build_parser"),
    "main": (".cli", "main"),
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
