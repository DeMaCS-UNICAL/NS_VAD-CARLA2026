from __future__ import annotations

import os
import shlex
import subprocess
import threading
from typing import BinaryIO

from ..logging.events import EventLogger
from .config import DpSrConfig


class _DpSrProcess:
    def __init__(
        self,
        *,
        config: DpSrConfig,
        dp_src_endpoint: tuple[str, int],
        logger: EventLogger,
    ) -> None:
        self._config = config
        self._dp_src_endpoint = dp_src_endpoint
        self._logger = logger
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_thread: threading.Thread | None = None

    @property
    def stdout(self) -> BinaryIO | None:
        if self._process is None:
            return None
        return self._process.stdout

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("DP-SR process is already running")

        self._config.check_runtime_requirements()
        command = self._config.build_command(dp_src_endpoint=self._dp_src_endpoint)
        stderr_target = str(self._logger.path)
        try:
            self._logger.reasoner("starting DP-SR process")
            self._debug(
                f"DP-SR process command={shlex.join(command)} stderr={stderr_target}"
            )
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self._config.jar_path.parent,
            )
            if self._process.stdout is None:
                raise RuntimeError("DP-SR process stdout pipe was not created")
            os.set_blocking(self._process.stdout.fileno(), False)
            if self._process.stderr is not None:
                self._stderr_thread = threading.Thread(
                    target=self._drain_stderr,
                    args=(self._process.stderr,),
                    name="dp-sr-stderr",
                    daemon=True,
                )
                self._stderr_thread.start()
        except Exception:
            self.stop()
            raise

        self._debug(f"spawned DP-SR process pid={self._process.pid}")

    def stop(self, terminate_timeout_s: float = 5.0) -> None:
        try:
            if self._process is not None:
                process_stdout = self._process.stdout
                if self._process.poll() is None:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=terminate_timeout_s)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait(timeout=1.0)
                if process_stdout is not None:
                    process_stdout.close()
                self._logger.reasoner("stopped DP-SR process")
        finally:
            self._process = None
            self._join_stderr_thread()

    def poll(self) -> int | None:
        if self._process is None:
            return None
        return self._process.poll()

    def ensure_running(self) -> None:
        return_code = self.poll()
        if self._process is not None and return_code is None:
            return

        if return_code is not None:
            self.wait_for_stderr()
        raise RuntimeError(
            "DP-SR process is not running"
            if return_code is None
            else (
                f"DP-SR exited with code {return_code}. "
                f"{self._details_hint()}"
            )
        )

    def wait_for_stderr(self) -> None:
        self._join_stderr_thread()

    def _drain_stderr(self, stderr: BinaryIO) -> None:
        try:
            for raw_line in iter(stderr.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line:
                    self._logger.dp_sr_log(line)
        finally:
            stderr.close()

    def _join_stderr_thread(self) -> None:
        if self._stderr_thread is None:
            return
        self._stderr_thread.join(timeout=1.0)
        self._stderr_thread = None

    def _details_hint(self) -> str:
        return f"Inspect {self._logger.path} for details."

    def _debug(self, message: str) -> None:
        if self._config.dpsr_debug:
            self._logger.reasoner(message)
