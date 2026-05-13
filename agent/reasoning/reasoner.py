from __future__ import annotations

import socket
import time
from types import TracebackType

from ..logging.events import EventLogger
from ..model.predicates import parse_reasoning_result
from ..model.types import PerceptionFrame, ReasoningResult
from .config import DpSrConfig
from .line_reader import _NonBlockingLineReader
from .process import _DpSrProcess
from .serialization import serialize_perception_frame
from .source_socket import _SourceSocket
from .state import _State


class DpSrReasoner:
    def __init__(
        self,
        *,
        config: DpSrConfig,
        logger: EventLogger,
    ) -> None:
        """Configure the DP-SR wrapper without starting the Java process."""
        self._config = config
        self._logger = logger

        self._process: _DpSrProcess | None = None
        self._stdout_reader: _NonBlockingLineReader | None = None
        self._source_socket: _SourceSocket | None = None
        self._state = _State.IDLE

    def __enter__(self) -> DpSrReasoner:
        """Connect the reasoner when entering a context manager block."""
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the reasoner when leaving a context manager block."""
        self.close()

    def connect(self) -> None:
        """Start DP-SR and wait until its source socket is connected."""
        if self._state is _State.CLOSED:
            raise RuntimeError("DP-SR reasoner is closed")
        if self._state is _State.CONNECTED:
            return

        try:
            dp_src_endpoint = self._open_source_socket()
            self._start_dp_sr(dp_src_endpoint)
            self._wait_until_ready(dp_src_endpoint=dp_src_endpoint)
            self._state = _State.CONNECTED
        except Exception:
            self._cleanup(reason="connect failed")
            raise

    def reason(self, frame: PerceptionFrame) -> list[ReasoningResult]:
        """Send one perception frame to DP-SR and return immediately available results."""
        self._ensure_connected()
        payload = serialize_perception_frame(frame)
        connection = self._require_source_connection()

        self._logger.dp_sr_input(payload)
        try:
            connection.sendall((payload + "\n").encode("utf-8"))
            return self._read_results(read_timeout_s=0.0)
        except OSError as exc:
            socket_context = self._describe_socket(connection)
            self._close_source_connection(reason="send failed")
            raise self._translate_transport_error(
                exc,
                socket_context=socket_context,
            ) from exc

    def drain(self, max_idle_wait_s: float = 5.00) -> list[ReasoningResult]:
        """Read results until stdout stays idle for max_idle_wait_s seconds."""
        self._ensure_connected()
        results: list[ReasoningResult] = []
        idle_deadline = time.monotonic() + max_idle_wait_s

        try:
            while time.monotonic() < idle_deadline:
                self._ensure_process_running()
                remaining = max(0.0, idle_deadline - time.monotonic())
                wait_step = min(self._config.read_timeout_s, remaining)
                new_results = self._read_results(read_timeout_s=wait_step)
                if not new_results:
                    continue
                results.extend(new_results)
                idle_deadline = time.monotonic() + max_idle_wait_s
        except OSError as exc:
            raise self._translate_transport_error(exc) from exc

        return results

    def close(self) -> None:
        """Stop DP-SR and release process, socket, listener, and log resources."""
        if self._state is _State.CLOSED:
            return

        self._state = _State.CLOSED
        self._cleanup(reason="reasoner closing")

    def _cleanup(self, *, reason: str = "closed") -> None:
        if self._state is not _State.CLOSED:
            self._state = _State.IDLE

        try:
            self._stop_process()
        finally:
            self._close_source_socket(reason=reason)

    def _open_source_socket(self) -> tuple[str, int]:
        self._source_socket = _SourceSocket()
        dp_src_endpoint = self._source_socket.endpoint
        self._debug(
            f"listening dp-src on {dp_src_endpoint[0]}:{dp_src_endpoint[1]}"
        )
        return dp_src_endpoint

    def _start_dp_sr(self, dp_src_endpoint: tuple[str, int]) -> None:
        self._process = _DpSrProcess(
            config=self._config,
            dp_src_endpoint=dp_src_endpoint,
            logger=self._logger,
        )

        try:
            self._process.start()
            if self._process.stdout is None:
                raise RuntimeError("DP-SR process stdout pipe was not created")

            self._stdout_reader = _NonBlockingLineReader(
                self._process.stdout,
                on_closed=lambda: self._debug("DP-SR stdout closed"),
            )
        except Exception:
            self._stop_process()
            raise

    def _read_results(self, read_timeout_s: float) -> list[ReasoningResult]:
        return [
            parse_reasoning_result(line)
            for line in self._read_available_lines(read_timeout_s=read_timeout_s)
        ]

    def _read_available_lines(self, read_timeout_s: float) -> list[str]:
        if self._stdout_reader is None:
            return []

        lines = self._stdout_reader.read_available_lines(timeout_s=read_timeout_s)
        for line in lines:
            self._logger.dp_sr_output(line)
        return lines

    def _wait_until_ready(
        self,
        *,
        dp_src_endpoint: tuple[str, int],
    ) -> None:
        if self._source_socket is None:
            raise RuntimeError("DP-SR source listener is not open")

        deadline = time.monotonic() + self._config.startup_timeout_s
        waiting_for_src_logged = False

        while self._source_socket.connection is None:
            if self._process is None:
                raise RuntimeError(
                    "DP-SR exited before establishing the source socket connection"
                )

            return_code = self._process.poll()
            if return_code is not None:
                self._process.wait_for_stderr()
                raise RuntimeError(
                    "DP-SR exited before establishing the source socket connection "
                    f"with code {return_code}. {self._details_hint()}"
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Timed out waiting for DP-SR source socket connection. "
                    f"{self._details_hint()}"
                )

            wait_step = min(0.10, remaining)
            if not waiting_for_src_logged:
                self._debug(
                    "waiting for DP-sr source connection on "
                    f"{dp_src_endpoint[0]}:{dp_src_endpoint[1]}"
                )
                waiting_for_src_logged = True
            self._accept_source_socket(wait_timeout_s=wait_step)

        self._logger.reasoner("DP-SR source socket connection established")
        self._debug(
            "DP-SR source socket connection established "
            f"dp-src={self._describe_socket(self._source_socket.connection)} "
        )

    def _accept_source_socket(self, wait_timeout_s: float) -> bool:
        if self._source_socket is None:
            raise RuntimeError("DP-SR source listener is not open")

        had_connection = self._source_socket.connection is not None
        accepted = self._source_socket.accept(timeout_s=wait_timeout_s)
        if accepted and not had_connection:
            self._debug(
                f"dp-src connected {self._describe_socket(self._source_socket.connection)}"
            )
        return accepted

    def _require_source_connection(self) -> socket.socket:
        if self._source_socket is None or self._source_socket.connection is None:
            raise RuntimeError("DP-SR source socket is not connected")
        return self._source_socket.connection

    def _close_source_connection(self, *, reason: str = "closed") -> None:
        if self._source_socket is None:
            return

        connection = self._source_socket.connection
        if connection is None:
            return

        self._debug(
            f"dp-src socket closed reason={reason} {self._describe_socket(connection)}"
        )
        self._source_socket.close_connection()

    def _close_source_socket(self, *, reason: str = "closed") -> None:
        if self._source_socket is None:
            return

        connection = self._source_socket.connection
        if connection is not None:
            self._debug(
                f"dp-src socket closed reason={reason} {self._describe_socket(connection)}"
            )
        self._source_socket.close()
        self._source_socket = None

    def _ensure_connected(self) -> None:
        if self._state is _State.CLOSED:
            raise RuntimeError("DP-SR reasoner is closed")
        if self._state is not _State.CONNECTED:
            raise RuntimeError("DP-SR reasoner is not connected")
        self._ensure_process_running()

    def _ensure_process_running(self) -> None:
        if self._process is None:
            raise RuntimeError("DP-SR process is not running")
        self._process.ensure_running()

    def _stop_process(self) -> None:
        try:
            if self._process is not None:
                self._process.stop()
        finally:
            self._process = None
            self._stdout_reader = None

    def _translate_transport_error(
        self,
        exc: OSError,
        *,
        socket_context: str | None = None,
    ) -> RuntimeError:
        return_code = None if self._process is None else self._process.poll()
        if return_code is not None and self._process is not None:
            self._process.wait_for_stderr()
        errno_text = "" if exc.errno is None else f" errno={exc.errno}"
        error_text = f"{exc.__class__.__name__}{errno_text}: {exc}"
        context_text = "" if socket_context is None else f" Socket: {socket_context}."
        if return_code is None:
            return RuntimeError(
                "DP-SR socket error while process is still running: "
                f"{error_text}.{context_text} {self._details_hint()}"
            )
        return RuntimeError(
            f"DP-SR closed its socket and exited with code {return_code}. "
            f"Last socket error: {error_text}.{context_text} "
            f"{self._details_hint()}"
        )

    def _details_hint(self) -> str:
        return f"Inspect {self._logger.path} for details."

    def _debug(self, message: str) -> None:
        if self._config.dpsr_debug:
            self._logger.reasoner(message)

    def _describe_socket(self, sock: socket.socket | None) -> str:
        if sock is None:
            return "socket=None"
        return (
            f"local={self._safe_socket_endpoint(sock, 'getsockname')} "
            f"remote={self._safe_socket_endpoint(sock, 'getpeername')}"
        )

    @staticmethod
    def _safe_socket_endpoint(sock: socket.socket, method_name: str) -> str:
        try:
            method = getattr(sock, method_name)
            host, port = method()
        except OSError:
            return "unknown"
        return f"{host}:{port}"
