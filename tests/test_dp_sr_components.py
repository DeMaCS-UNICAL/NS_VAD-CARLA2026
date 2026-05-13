from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent.logging.events import EventLogger
from agent.model.types import PerceptionFrame
from agent.reasoning.line_reader import _NonBlockingLineReader
from agent.reasoning.reasoner import DpSrReasoner
from agent.reasoning.serialization import serialize_perception_frame
from agent.reasoning.config import DpSrConfig, PACKAGED_DP_SR_JAR_PATH
from agent.reasoning.source_socket import _SourceSocket
from agent.reasoning.state import _State


def make_config() -> DpSrConfig:
    return DpSrConfig(
        rules_path="unused.lp",
        read_timeout_s=0.02,
        startup_timeout_s=10.0,
        dpsr_debug=False,
    )


class ReasonerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.logger = EventLogger(str(Path(self._tmp_dir.name) / "agent.log"))
        self.addCleanup(self.logger.close)

    def make_reasoner(self) -> DpSrReasoner:
        return DpSrReasoner(
            config=make_config(),
            logger=self.logger,
        )


class SourceSocketTests(unittest.TestCase):
    def test_endpoint_returns_host_and_port(self) -> None:
        source_socket = _SourceSocket()
        self.addCleanup(source_socket.close)

        host, port = source_socket.endpoint

        self.assertEqual("127.0.0.1", host)
        self.assertGreater(port, 0)

    def test_accept_timeout_returns_false(self) -> None:
        source_socket = _SourceSocket()
        self.addCleanup(source_socket.close)

        self.assertFalse(source_socket.accept(timeout_s=0.0))

    def test_close_does_not_raise(self) -> None:
        source_socket = _SourceSocket()

        source_socket.close()
        source_socket.close()


class DpSrConfigTests(unittest.TestCase):
    def test_config_uses_packaged_dp_sr_jar(self) -> None:
        config = DpSrConfig(
            rules_path="rules.lp",
            read_timeout_s=0.02,
            startup_timeout_s=10.0,
            dpsr_debug=False,
        )

        self.assertEqual(PACKAGED_DP_SR_JAR_PATH, config.jar_path)
        self.assertEqual(Path("rules.lp").resolve(), config.rules_path)
        self.assertIn(
            str(PACKAGED_DP_SR_JAR_PATH),
            config.build_command(dp_src_endpoint=("127.0.0.1", 12345)),
        )


class SerializationTests(unittest.TestCase):
    def test_serialize_perception_frame_preserves_dp_sr_payload_format(self) -> None:
        self.assertEqual(
            "7 object(car_1,car,10,20); speed(car_1,fast);",
            serialize_perception_frame(
                PerceptionFrame(
                    frame_id=7,
                    facts=("object(car_1,car,10,20)", "speed(car_1,fast)"),
                )
            ),
        )

    def test_serialize_empty_perception_frame_preserves_trailing_separator(self) -> None:
        self.assertEqual(
            "7 ;",
            serialize_perception_frame(PerceptionFrame(frame_id=7, facts=())),
        )


class NonBlockingLineReaderTests(unittest.TestCase):
    def test_read_available_lines_preserves_partial_line_between_reads(self) -> None:
        read_fd, write_fd = os.pipe()
        stream = os.fdopen(read_fd, "rb", buffering=0)
        try:
            reader = _NonBlockingLineReader(stream)

            os.write(write_fd, b"first\nsec")
            self.assertEqual(["first"], reader.read_available_lines(timeout_s=0.0))

            os.write(write_fd, b"ond\n")
            self.assertEqual(["second"], reader.read_available_lines(timeout_s=0.0))
        finally:
            os.close(write_fd)
            stream.close()


class DpSrReasonerStateTests(ReasonerTestCase):
    def test_connect_failure_cleans_up_without_closing_reasoner(self) -> None:
        reasoner = self.make_reasoner()
        frame = PerceptionFrame(frame_id=1, facts=())

        try:
            with mock.patch.object(
                DpSrReasoner,
                "_start_dp_sr",
                side_effect=RuntimeError("boom"),
            ):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    reasoner.connect()

            self.assertIs(reasoner._state, _State.IDLE)
            self.assertIsNone(reasoner._source_socket)
            with self.assertRaisesRegex(RuntimeError, "DP-SR reasoner is not connected"):
                reasoner.reason(frame)
        finally:
            reasoner.close()

    def test_close_marks_reasoner_closed(self) -> None:
        reasoner = self.make_reasoner()

        reasoner.close()

        self.assertIs(reasoner._state, _State.CLOSED)

    def test_reason_before_connect_raises_not_connected(self) -> None:
        reasoner = self.make_reasoner()
        try:
            with self.assertRaisesRegex(RuntimeError, "DP-SR reasoner is not connected"):
                reasoner.reason(PerceptionFrame(frame_id=1, facts=()))
        finally:
            reasoner.close()

    def test_reason_after_close_raises_closed(self) -> None:
        reasoner = self.make_reasoner()

        reasoner.close()

        with self.assertRaisesRegex(RuntimeError, "DP-SR reasoner is closed"):
            reasoner.reason(PerceptionFrame(frame_id=1, facts=()))


if __name__ == "__main__":
    unittest.main()
