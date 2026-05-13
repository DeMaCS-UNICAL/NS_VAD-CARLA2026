from __future__ import annotations

import queue
import threading
import time
import traceback

import cv2

from ..logging.events import (
    EventLogger,
    format_candidate_response,
)
from ..model.types import CandidateAnomaly, VLMResult
from .config import VLMConfig
from .media.extractor import ClipExtractor
from .media.saver import VlmInputSaver
from .models import ExtractedClip
from .protocols import VisualLanguageClassifier


class VisualReasoner:
    """Asynchronous VLM worker pool for candidate anomaly validation."""

    def __init__(
        self,
        *,
        config: VLMConfig,
        classifier: VisualLanguageClassifier,
        logger: EventLogger,
    ) -> None:
        """Start VLM workers using a shared classifier and extraction config."""
        self._vlm_input_mode = config.vlm_input_mode
        self._logger = logger
        self._worker_join_timeout_s = config.worker_join_timeout_s

        self._clip_extractor = ClipExtractor(
            video_path=str(config.video_path),
            max_frames_per_clip=config.max_frames_per_clip,
        )
        self._input_saver = VlmInputSaver(
            output_dir=config.output_dir,
            vlm_input_mode=config.vlm_input_mode,
            enabled=config.save_vlm_input,
            logger=logger,
        )

        self._task_queue: queue.Queue[CandidateAnomaly | None] = queue.Queue()
        self._state_condition = threading.Condition()

        self._accepting_tasks = True
        self._closed = False
        self._submitted_candidates = 0
        self._completed_candidates = 0
        self._failed_candidates = 0

        self._classifier = classifier
        self._classifier_lock = threading.Lock()

        self._threads = [
            threading.Thread(
                target=self._worker_loop,
                name=f"vlm-worker-{index}",
                daemon=True,
            )
            for index in range(config.max_workers)
        ]

        for thread in self._threads:
            thread.start()

    def submit_candidate(self, candidate: CandidateAnomaly) -> bool:
        """Queue a candidate for VLM validation if the service is accepting work."""
        with self._state_condition:
            if self._closed or not self._accepting_tasks:
                return False

            self._task_queue.put_nowait(candidate)
            self._submitted_candidates += 1

        self._logger.vlm_input(
            f"{self._candidate_context(candidate)} event=candidate_submitted"
        )
        return True

    @property
    def failed_candidates(self) -> int:
        """Number of candidates that failed during asynchronous VLM processing."""
        with self._state_condition:
            return self._failed_candidates

    def stop_accepting(self) -> None:
        """Stop accepting new candidates while allowing queued work to finish."""
        with self._state_condition:
            self._accepting_tasks = False

    def flush_pending(self, timeout_s: float = 120.0) -> bool:
        """Wait until queued candidates are processed or the timeout expires."""
        deadline = time.perf_counter() + timeout_s

        with self._state_condition:
            while self._completed_candidates < self._submitted_candidates:
                remaining_s = deadline - time.perf_counter()

                if remaining_s <= 0:
                    break

                self._state_condition.wait(timeout=remaining_s)

            pending_candidates = (
                self._submitted_candidates - self._completed_candidates
            )
            flushed = pending_candidates == 0

        if not flushed:
            self._logger.warning(
                (
                    "source=vlm "
                    "event=flush_timeout "
                    f"timeout_s={timeout_s:.1f} "
                    f"pending_candidates={pending_candidates}"
                ),
                console=True,
            )

        return flushed

    def close(self) -> None:
        """Signal workers to stop, join them, and close the shared classifier."""
        with self._state_condition:
            if self._closed:
                return

            self._closed = True
            self._accepting_tasks = False

        for _ in self._threads:
            self._task_queue.put_nowait(None)

        for thread in self._threads:
            thread.join(timeout=self._worker_join_timeout_s)

            if thread.is_alive():
                self._logger.warning(
                    (
                        "source=vlm "
                        "event=worker_join_timeout "
                        f"worker={thread.name}"
                    ),
                    console=True,
                )

        self._close_classifier(self._classifier)

    def _worker_loop(self) -> None:
        capture: cv2.VideoCapture | None = None

        try:
            while True:
                candidate = self._task_queue.get()

                try:
                    if candidate is None:
                        return

                    if capture is None:
                        capture = self._clip_extractor.open_capture()

                    result = self._process_candidate(
                        candidate=candidate,
                        capture=capture,
                        classifier=self._classifier,
                    )

                    self._log_candidate_response(result)

                except Exception as exc:
                    if candidate is not None:
                        self._record_failure()
                    self._log_worker_error(exc, candidate)

                finally:
                    if candidate is not None:
                        self._record_completion()

                    self._task_queue.task_done()

        finally:
            self._release_capture(capture)

    def _process_candidate(
        self,
        *,
        candidate: CandidateAnomaly,
        capture: cv2.VideoCapture,
        classifier: VisualLanguageClassifier,
    ) -> VLMResult:
        clip = self._clip_extractor.extract_clip(
            capture=capture,
            start_frame_id=candidate.start_frame_id,
            end_frame_id=candidate.end_frame_id,
        )

        self._input_saver.save(candidate, clip)
        self._log_clip_summary(candidate, clip)

        return self._classify_candidate(classifier, candidate, clip)

    def _log_clip_summary(
        self,
        candidate: CandidateAnomaly,
        clip: ExtractedClip,
    ) -> None:
        message = (
            f"{self._candidate_context(candidate)} "
            "event=clip_extracted "
            f"sampled_frames={clip.sampled_frames}"
        )

        if self._vlm_input_mode != "frames":
            message += f" input_mode={self._vlm_input_mode}"

        self._logger.vlm_input(message)

    def _log_candidate_response(self, result: VLMResult) -> None:
        self._logger.candidate_response(
            format_candidate_response(result),
            anomalous=result.anomaly,
            console=True,
        )

    def _log_worker_error(
        self,
        exc: Exception,
        candidate: CandidateAnomaly | None,
    ) -> None:
        context = (
            "source=vlm"
            if candidate is None
            else self._candidate_context(candidate)
        )

        self._logger.error(
            (
                f"{context} "
                "event=worker_error "
                f"error={exc} "
                f"traceback={traceback.format_exc()}"
            ),
            console=True,
        )

    def _candidate_context(self, candidate: CandidateAnomaly) -> str:
        return (
            "source=vlm "
            f"type={candidate.type} "
            f"id={candidate.id} "
            f"frame_range={candidate.start_frame_id}-{candidate.end_frame_id}"
        )

    def _classify_candidate(
        self,
        classifier: VisualLanguageClassifier,
        candidate: CandidateAnomaly,
        clip: ExtractedClip,
    ) -> VLMResult:
        with self._classifier_lock:
            return classifier(candidate, clip)

    def _record_completion(self) -> None:
        with self._state_condition:
            self._completed_candidates += 1
            self._state_condition.notify_all()

    def _record_failure(self) -> None:
        with self._state_condition:
            self._failed_candidates += 1

    @staticmethod
    def _release_capture(capture: cv2.VideoCapture | None) -> None:
        if capture is not None:
            capture.release()

    @staticmethod
    def _close_classifier(classifier: VisualLanguageClassifier | None) -> None:
        if classifier is None:
            return

        close_method = getattr(classifier, "close", None)

        if callable(close_method):
            close_method()
