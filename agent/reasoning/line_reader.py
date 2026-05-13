from __future__ import annotations

import os
import select
from collections.abc import Callable
from typing import BinaryIO


class _NonBlockingLineReader:
    def __init__(
        self,
        stream: BinaryIO,
        *,
        on_closed: Callable[[], None] | None = None,
    ) -> None:
        self._stream = stream
        self._buffer = ""
        self._on_closed = on_closed

    def read_available_lines(self, timeout_s: float) -> list[str]:
        if self._stream.closed:
            return []

        select_timeout = timeout_s
        while True:
            ready, _, _ = select.select([self._stream], [], [], select_timeout)
            if not ready:
                break

            try:
                chunk = os.read(self._stream.fileno(), 65535)
            except (BlockingIOError, InterruptedError):
                break

            if not chunk:
                if self._on_closed is not None:
                    self._on_closed()
                self._stream.close()
                break

            self._buffer += chunk.decode("utf-8", errors="replace")
            select_timeout = 0.0

        return self._pop_buffered_lines()

    def _pop_buffered_lines(self) -> list[str]:
        lines: list[str] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
        return lines
