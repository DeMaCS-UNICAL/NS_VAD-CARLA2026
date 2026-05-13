from __future__ import annotations

import re


ATOM_PATTERN = re.compile(r"[^a-zA-Z0-9_]")


def sanitize_atom(value: object) -> str:
    text = str(value).strip().lower()
    text = ATOM_PATTERN.sub("_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        return "unknown"
    if text[0].isdigit():
        return f"v_{text}"
    return text
