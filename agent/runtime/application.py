from __future__ import annotations

from typing import TYPE_CHECKING

from ..logging.events import (
    EventLogger,
    format_anomaly,
    format_candidate_anomaly,
)
from ..model.types import ReasoningResult

if TYPE_CHECKING:
    from ..reasoning.reasoner import DpSrReasoner
    from ..vision.perception import Perception
    from ..vlm import VisualReasoner


class Agent:
    def __init__(
        self,
        *,
        perception: Perception,
        reasoner: DpSrReasoner,
        visual_language_model: VisualReasoner,
        logger: EventLogger,
        log_facts: bool = False,
    ) -> None:
        self._log_facts_enabled = log_facts
        self._logger = logger
        self._seen_candidates: set[tuple[str, str, int, int]] = set()
        self.perception = perception
        self.reasoner = reasoner
        self.visual_language_model = visual_language_model

    def run(self) -> None:
        self._logger.info("Agent started", console=True)
        try:
            while not self.perception.has_ended():
                try:
                    frame = self.perception.next_frame()
                except StopIteration:
                    break

                if self._log_facts_enabled:
                    serialized_facts = "; ".join(frame.facts)
                    self._logger.perception(
                        (
                            f"frame_id={frame.frame_id} "
                            f"facts_count={len(frame.facts)} "
                            f"facts={serialized_facts}"
                        ),
                        console=True,
                        persist=False,
                    )
                for result in self.reasoner.reason(frame):
                    self._handle_reasoning_result(result)

            for pending in self.reasoner.drain():
                self._handle_reasoning_result(pending)
        finally:
            self.visual_language_model.stop_accepting()
            self.visual_language_model.flush_pending(timeout_s=30.0)
            self.perception.close()
            self.reasoner.close()
            self.visual_language_model.close()

    def _handle_reasoning_result(self, result: ReasoningResult) -> None:
        for anomaly in result.anomalies:
            self._logger.anomaly(
                format_anomaly(anomaly),
                console=True,
            )

        for candidate in result.candidate_anomalies:
            candidate_key = (
                candidate.id,
                candidate.type,
                candidate.start_frame_id,
                candidate.end_frame_id,
            )
            if candidate_key not in self._seen_candidates:
                self._seen_candidates.add(candidate_key)
                self._logger.candidate(
                    format_candidate_anomaly(candidate),
                    console=True,
                )
                self.visual_language_model.submit_candidate(candidate)
