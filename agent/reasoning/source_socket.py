from __future__ import annotations

import select
import socket


class _SourceSocket:
    """Manages the dp-src socket listener and connection."""
    def __init__(self, hostname: str = "127.0.0.1") -> None:
        self._listener: socket.socket | None = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        )
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((hostname, 0))
        self._listener.listen(1)
        self._connection: socket.socket | None = None

    @property
    def endpoint(self) -> tuple[str, int]:
        if self._listener is None:
            raise RuntimeError("DP-SR source listener is not open")
        host, port = self._listener.getsockname()
        return str(host), int(port)

    @property
    def connection(self) -> socket.socket | None:
        return self._connection

    def accept(self, timeout_s: float) -> bool:
        if self._connection is not None:
            return True
        if self._listener is None:
            raise RuntimeError("DP-SR source listener is not open")

        ready, _, _ = select.select([self._listener], [], [], timeout_s)
        if not ready:
            return False

        self._connection, _src_addr = self._listener.accept()
        return True

    def send(self, data: bytes) -> None:
        if self._connection is None:
            raise RuntimeError("DP-SR source socket is not connected")
        self._connection.sendall(data)

    def close_connection(self) -> socket.socket | None:
        connection = self._connection
        if connection is None:
            return None

        try:
            connection.close()
            return connection
        finally:
            self._connection = None

    def close(self) -> None:
        self.close_connection()
        if self._listener is None:
            return
        try:
            self._listener.close()
        finally:
            self._listener = None
