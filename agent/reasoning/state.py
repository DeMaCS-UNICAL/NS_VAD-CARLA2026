from __future__ import annotations

from enum import Enum, auto


class _State(Enum):
    IDLE = auto()
    CONNECTED = auto()
    CLOSED = auto()
