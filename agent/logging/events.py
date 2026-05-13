from __future__ import annotations

import logging
import sys
import threading
from itertools import count
from pathlib import Path
from typing import TextIO

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from rich.theme import Theme

from ..model.types import Anomaly, CandidateAnomaly, VLMResult


_LOGGER_COUNTER = count()
_RECORD_PERSIST_ATTR = "agent_persist"
_RECORD_CONSOLE_ATTR = "agent_console"
_RECORD_STYLE_ATTR = "agent_style"

_STYLE_AGENT = "agent.agent"
_STYLE_INFO = "agent.info"
_STYLE_PERCEPTION = "agent.perception"
_STYLE_REASONER = "agent.reasoner"
_STYLE_DP_SR = "agent.dp_sr"
_STYLE_VLM = "agent.vlm"
_STYLE_CANDIDATE = "agent.candidate"
_STYLE_ANOMALY = "agent.anomaly"
_STYLE_CANDIDATE_RESPONSE_CLEAR = "agent.candidate_response_clear"
_STYLE_CANDIDATE_RESPONSE_ANOMALOUS = "agent.candidate_response_anomalous"
_STYLE_WARNING = "agent.warning"
_STYLE_ERROR = "agent.error"

_CHANNELS: dict[str, tuple[str, str, int]] = {
    "agent": ("[agent]", _STYLE_AGENT, logging.INFO),
    "info": ("[info]", _STYLE_INFO, logging.INFO),
    "perception": ("[perception]", _STYLE_PERCEPTION, logging.INFO),
    "reasoner": ("[reasoner]", _STYLE_REASONER, logging.INFO),
    "dp_sr_input": ("[agent->dp-sr]", _STYLE_DP_SR, logging.INFO),
    "dp_sr_output": ("[dp-sr->agent]", _STYLE_DP_SR, logging.INFO),
    "dp_sr_log": ("[dp-sr]", _STYLE_DP_SR, logging.INFO),
    "vlm_input": ("[vlm-input]", _STYLE_VLM, logging.INFO),
    "candidate": ("[candidate]", _STYLE_CANDIDATE, logging.INFO),
    "anomaly": ("[anomaly]", _STYLE_ANOMALY, logging.WARNING),
    "warning": ("[warning]", _STYLE_WARNING, logging.WARNING),
    "error": ("[error]", _STYLE_ERROR, logging.ERROR),
}

_PREFIX_STYLES = {
    prefix: style
    for prefix, style, _ in _CHANNELS.values()
}
_PREFIX_STYLES["[candidate-response]"] = _STYLE_CANDIDATE_RESPONSE_CLEAR

_PREFIX_LEVELS = {
    prefix: level
    for prefix, _, level in _CHANNELS.values()
    if level != logging.INFO
}

_CONSOLE_THEME = Theme(
    {
        _STYLE_AGENT: "cyan",
        _STYLE_INFO: "cyan",
        _STYLE_PERCEPTION: "dim",
        _STYLE_REASONER: "dim cyan",
        _STYLE_DP_SR: "dim",
        _STYLE_VLM: "magenta",
        _STYLE_CANDIDATE: "yellow",
        _STYLE_ANOMALY: "bold red",
        _STYLE_CANDIDATE_RESPONSE_CLEAR: "green",
        _STYLE_CANDIDATE_RESPONSE_ANOMALOUS: "bold red",
        _STYLE_WARNING: "yellow",
        _STYLE_ERROR: "bold red",
    }
)


class _RouteFilter(logging.Filter):
    def __init__(self, route_attr: str) -> None:
        super().__init__()
        self._route_attr = route_attr

    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, self._route_attr, False))


class _AgentRichHandler(RichHandler):
    def render_message(
        self,
        record: logging.LogRecord,
        message: str,
    ) -> Text:
        message_text = Text(message)

        style = getattr(record, _RECORD_STYLE_ATTR, None)
        if style:
            message_text.stylize(style)

        highlighter = getattr(record, "highlighter", self.highlighter)
        if highlighter:
            message_text = highlighter(message_text)

        return message_text


class EventLogger:
    def __init__(self, log_path: str, console_stream: TextIO | None = None) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._console_stream = console_stream or sys.stdout
        self._lock = threading.Lock()
        self._closed = False

        self._logger = logging.getLogger(f"agent.events.{next(_LOGGER_COUNTER)}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        self._handlers: list[logging.Handler] = []
        self._configure_handlers()

    @property
    def path(self) -> Path:
        return self._path

    def __enter__(self) -> EventLogger:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return

            for handler in self._handlers:
                self._logger.removeHandler(handler)
                handler.close()

            self._handlers.clear()
            self._closed = True

    def agent(
        self,
        message: str = "",
        *,
        console: bool = False,
        persist: bool = True,
    ) -> None:
        self._emit_channel("agent", message, console=console, persist=persist)

    def info(
        self,
        message: str = "",
        *,
        console: bool = False,
        persist: bool = True,
    ) -> None:
        self._emit_channel("info", message, console=console, persist=persist)

    def perception(
        self,
        message: str = "",
        *,
        console: bool = False,
        persist: bool = True,
    ) -> None:
        self._emit_channel("perception", message, console=console, persist=persist)

    def reasoner(
        self,
        message: str = "",
        *,
        console: bool = False,
        persist: bool = True,
    ) -> None:
        self._emit_channel("reasoner", message, console=console, persist=persist)

    def dp_sr_input(
        self,
        message: str = "",
        *,
        console: bool = False,
        persist: bool = True,
    ) -> None:
        self._emit_channel("dp_sr_input", message, console=console, persist=persist)

    def dp_sr_output(
        self,
        message: str = "",
        *,
        console: bool = False,
        persist: bool = True,
    ) -> None:
        self._emit_channel("dp_sr_output", message, console=console, persist=persist)

    def dp_sr_log(
        self,
        message: str = "",
        *,
        console: bool = False,
        persist: bool = True,
    ) -> None:
        self._emit_channel("dp_sr_log", message, console=console, persist=persist)

    def vlm_input(
        self,
        message: str = "",
        *,
        console: bool = False,
        persist: bool = True,
    ) -> None:
        self._emit_channel("vlm_input", message, console=console, persist=persist)

    def candidate(
        self,
        message: str = "",
        *,
        console: bool = False,
        persist: bool = True,
    ) -> None:
        self._emit_channel("candidate", message, console=console, persist=persist)

    def anomaly(
        self,
        message: str = "",
        *,
        console: bool = False,
        persist: bool = True,
    ) -> None:
        self._emit_channel("anomaly", message, console=console, persist=persist)

    def warning(
        self,
        message: str = "",
        *,
        console: bool = False,
        persist: bool = True,
    ) -> None:
        self._emit_channel("warning", message, console=console, persist=persist)

    def error(
        self,
        message: str = "",
        *,
        console: bool = False,
        persist: bool = True,
    ) -> None:
        self._emit_channel("error", message, console=console, persist=persist)

    def candidate_response(
        self,
        message: str = "",
        *,
        anomalous: bool,
        console: bool = False,
        persist: bool = True,
    ) -> None:
        style = (
            _STYLE_CANDIDATE_RESPONSE_ANOMALOUS
            if anomalous
            else _STYLE_CANDIDATE_RESPONSE_CLEAR
        )

        self._emit(
            "[candidate-response]",
            message,
            console=console,
            persist=persist,
            style=style,
            level=logging.WARNING if anomalous else logging.INFO,
        )

    def log(
        self,
        prefix: str,
        message: str = "",
        *,
        console: bool = False,
        console_message: str | None = None,
        persist: bool = True,
    ) -> None:
        self._emit(
            prefix,
            message,
            console=console,
            console_message=console_message,
            persist=persist,
            style=_PREFIX_STYLES.get(prefix),
            level=_PREFIX_LEVELS.get(prefix, logging.INFO),
        )

    def _emit_channel(
        self,
        channel: str,
        message: str = "",
        *,
        console: bool,
        persist: bool,
    ) -> None:
        prefix, style, level = _CHANNELS[channel]

        self._emit(
            prefix,
            message,
            console=console,
            persist=persist,
            style=style,
            level=level,
        )

    def _configure_handlers(self) -> None:
        file_handler = logging.FileHandler(
            self._path,
            mode="w",
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        file_handler.addFilter(_RouteFilter(_RECORD_PERSIST_ATTR))

        console_is_terminal = _stream_is_terminal(self._console_stream)
        console = Console(
            file=self._console_stream,
            force_terminal=console_is_terminal,
            color_system="standard" if console_is_terminal else None,
            theme=_CONSOLE_THEME,
            soft_wrap=True,
        )

        console_handler = _AgentRichHandler(
            console=console,
            show_time=False,
            show_level=False,
            show_path=False,
            enable_link_path=False,
            rich_tracebacks=True,
            highlighter=None,
            markup=False,
        )
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        console_handler.addFilter(_RouteFilter(_RECORD_CONSOLE_ATTR))

        self._handlers = [file_handler, console_handler]

        for handler in self._handlers:
            self._logger.addHandler(handler)

    def _emit(
        self,
        prefix: str,
        message: str = "",
        *,
        console: bool,
        persist: bool,
        style: str | None,
        console_message: str | None = None,
        level: int = logging.INFO,
    ) -> None:
        line = _format_log_line(prefix, message)

        if console_message is None or console_message == line:
            self._write(
                line,
                console=console,
                persist=persist,
                style=style,
                level=level,
            )
            return

        if persist:
            self._write(
                line,
                console=False,
                persist=True,
                style=style,
                level=level,
            )

        if console:
            self._write(
                console_message,
                console=True,
                persist=False,
                style=style,
                level=level,
            )

    def _write(
        self,
        line: str,
        *,
        console: bool,
        persist: bool,
        style: str | None,
        level: int,
    ) -> None:
        if not console and not persist:
            return

        with self._lock:
            self._ensure_open()

            self._logger.log(
                level,
                line,
                extra={
                    _RECORD_CONSOLE_ATTR: console,
                    _RECORD_PERSIST_ATTR: persist,
                    _RECORD_STYLE_ATTR: style,
                },
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("EventLogger is closed")


def format_candidate_anomaly(candidate: CandidateAnomaly) -> str:
    return _format_typed_object_event(
        id=candidate.id,
        event_type=candidate.type,
        start_frame_id=candidate.start_frame_id,
        end_frame_id=candidate.end_frame_id,
    )


def format_anomaly(anomaly: Anomaly) -> str:
    return _format_typed_object_event(
        id=anomaly.id,
        event_type=anomaly.type,
        start_frame_id=anomaly.start_frame_id,
        end_frame_id=anomaly.end_frame_id,
    )


def format_candidate_response(result: VLMResult) -> str:
    status = "anomalous" if result.anomaly else "not anomalous"

    event = _format_typed_object_event(
        id=result.candidate.id,
        event_type=result.candidate.type,
        start_frame_id=result.candidate.start_frame_id,
        end_frame_id=result.candidate.end_frame_id,
    )

    return f"{event} {status}. Reason: {result.reason}"


def _format_typed_object_event(
    id: str,
    event_type: str,
    start_frame_id: int,
    end_frame_id: int,
) -> str:
    return _format_event_with_time_range(
        subject=_format_object_subject(id, event_type),
        start_frame_id=start_frame_id,
        end_frame_id=end_frame_id,
    )


def _format_object_subject(id: str, event_type: str) -> str:
    return f"Entity {id} {event_type}"


def _format_event_with_time_range(
    subject: str,
    start_frame_id: int,
    end_frame_id: int,
) -> str:
    if start_frame_id == end_frame_id:
        return f"{subject} at frame {start_frame_id}"

    return f"{subject} from frame {start_frame_id} to frame {end_frame_id}"


def _format_log_line(prefix: str, message: str) -> str:
    return prefix if not message else f"{prefix} {message}"


def _stream_is_terminal(stream: TextIO) -> bool:
    isatty = getattr(stream, "isatty", None)

    if not callable(isatty):
        return False

    try:
        return bool(isatty())
    except OSError:
        return False
