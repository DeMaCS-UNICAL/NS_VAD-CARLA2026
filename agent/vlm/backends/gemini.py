from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    genai = None
    genai_types = None

import cv2

from ...model.types import CandidateAnomaly, VLMResult
from ..config import SUPPORTED_VLM_INPUT_MODES
from ..media.writer import write_clip_video
from ..protocols import VisualLanguageClassifier
from .gemini_payload import (
    GEMINI_GENERATION_CONFIG,
    build_gemini_prompt,
    parse_gemini_response,
)

if TYPE_CHECKING:
    from ..models import ExtractedClip


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
INLINE_VIDEO_MAX_BYTES = 20 * 1024 * 1024
VIDEO_MIME_TYPE = "video/mp4"


def _build_genai_client(api_key: str) -> genai.Client:
    """Create and return a Google GenAI client using the provided API key.

    Raises:
        RuntimeError: If the google-genai dependency is not installed.
    """
    if genai is None:
        raise RuntimeError(
            "google-genai is not installed. Install it to use the Gemini VLM backend."
        )
    return genai.Client(api_key=api_key)


class GeminiVLMClassifier(VisualLanguageClassifier):
    """Visual-language classifier backed by the Gemini API.

    The classifier receives a candidate anomaly and an extracted video clip,
    builds a Gemini-compatible request payload, sends it to the selected Gemini
    model, and converts the model response into a VLMResult.

    Depending on the selected input mode, the clip is sent either as individual
    JPEG frames or as an MP4 video. Large videos are uploaded through the Gemini
    Files API, while smaller videos are sent inline.
    """
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_GEMINI_MODEL,
        input_mode: str = "video",
        timeout_s: float = 30.0,
        inline_video_max_bytes: int = INLINE_VIDEO_MAX_BYTES,
    ) -> None:
        """Initialize the Gemini VLM classifier.

        Args:
            api_key: Gemini API key used to authenticate requests.
            model: Gemini model name to use for classification.
            input_mode: Input representation sent to Gemini. Supported values are
                "video" and "frames".
            timeout_s: Maximum time to wait for uploaded video processing.
            inline_video_max_bytes: Maximum video size, in bytes, that can be
                sent inline instead of using the Files API.

        Raises:
            ValueError: If the API key, model name, or input mode is invalid.
        """
        normalized_api_key = api_key.strip()
        if not normalized_api_key:
            raise ValueError("Gemini API key cannot be empty")

        normalized_model = model.strip()
        if not normalized_model:
            raise ValueError("Gemini model cannot be empty")
        if input_mode not in SUPPORTED_VLM_INPUT_MODES:
            raise ValueError(f"Unsupported Gemini VLM input mode: {input_mode}")

        self._api_key = normalized_api_key
        self._model = normalized_model
        self._input_mode = input_mode
        self._timeout_s = timeout_s
        self._inline_video_max_bytes = inline_video_max_bytes
        self._client = _build_genai_client(api_key=normalized_api_key)

    def __call__(
        self,
        candidate_anomaly: CandidateAnomaly,
        clip: ExtractedClip,
    ) -> VLMResult:
        """Classify a candidate anomaly using Gemini.

        Args:
            candidate_anomaly: Candidate anomaly produced by the symbolic layer.
            clip: Extracted clip associated with the candidate anomaly.

        Returns:
            A VLMResult containing the original candidate, the anomaly decision,
            and the explanation returned by the model.
        """
        if not clip.frames:
            return VLMResult(
                candidate=candidate_anomaly,
                anomaly=False,
                reason="No frames extracted for VLM analysis",
            )

        payload, cleanup = self._build_request_payload(candidate_anomaly, clip)
        try:
            response_payload = self._generate_content(payload)
        finally:
            cleanup()
        parsed_response = parse_gemini_response(response_payload)
        return VLMResult(
            candidate=candidate_anomaly,
            anomaly=parsed_response["anomalous"],
            reason=parsed_response["reason"],
        )

    def _build_request_payload(
        self,
        candidate_anomaly: CandidateAnomaly,
        clip: ExtractedClip,
    ) -> tuple[list[Any], Callable[[], None]]:
        """Build the Gemini request payload for a candidate anomaly.

        The returned payload always starts with the textual prompt. Depending on
        the configured input mode, the visual evidence is then added either as
        an encoded video part or as a sequence of encoded frame parts.

        Args:
            candidate_anomaly: Candidate anomaly to describe in the prompt.
            clip: Extracted clip containing the visual evidence.

        Returns:
            A tuple containing the Gemini payload parts and a cleanup callback.
            The cleanup callback removes any remote resource created during
            payload construction, such as uploaded Gemini files.
        """
        parts: list[Any] = [
            build_gemini_prompt(
                candidate_anomaly=candidate_anomaly,
                sampled_frames=clip.sampled_frames,
                input_mode=self._input_mode,
            )
        ]
        if self._input_mode == "video":
            video_part, cleanup = self._encode_video_part(clip)
            parts.append(video_part)
            return parts, cleanup

        parts.extend(self._encode_frame_part(frame) for frame in clip.frames)
        return parts, _no_cleanup_required

    def _encode_frame_part(self, frame: object) -> genai_types.Part:
        """Encode a single frame as a Gemini inline JPEG part.

        Args:
            frame: OpenCV frame to encode.

        Returns:
            A Gemini Part containing the encoded JPEG bytes.

        Raises:
            RuntimeError: If the google-genai dependency is missing or the frame
            cannot be encoded.
        """
        if genai_types is None:
            raise RuntimeError(
                "google-genai is not installed. Install it to use the Gemini VLM backend."
            )
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 85],
        )
        if not ok:
            raise RuntimeError("Failed to encode frame for Gemini request")

        return genai_types.Part.from_bytes(
            data=encoded.tobytes(),
            mime_type="image/jpeg",
        )

    def _encode_video_part(
        self,
        clip: ExtractedClip,
    ) -> tuple[Any, Callable[[], None]]:
        """Encode an extracted clip as a Gemini video part.

        The clip is first written to a temporary MP4 file. If the resulting file
        is small enough, it is sent inline. Otherwise, it is uploaded through the
        Gemini Files API.

        Args:
            clip: Extracted clip to encode.

        Returns:
            A tuple containing the Gemini video part and a cleanup callback.

        Raises:
            RuntimeError: If the google-genai dependency is missing or the video
            cannot be encoded/uploaded.
        """
        if genai_types is None:
            raise RuntimeError(
                "google-genai is not installed. Install it to use the Gemini VLM backend."
            )

        video_path = self._write_temp_video(clip)
        try:
            video_size = Path(video_path).stat().st_size
            if video_size <= self._inline_video_max_bytes:
                video_bytes = Path(video_path).read_bytes()
                return (
                    genai_types.Part(
                        inline_data=genai_types.Blob(
                            data=video_bytes,
                            mime_type=VIDEO_MIME_TYPE,
                        ),
                        video_metadata=genai_types.VideoMetadata(
                            fps=clip.playback_fps,
                        ),
                    ),
                    _no_cleanup_required,
                )
            return self._upload_video_part(video_path=video_path, fps=clip.playback_fps)
        finally:
            Path(video_path).unlink(missing_ok=True)

    def _upload_video_part(
        self,
        *,
        video_path: str,
        fps: float,
    ) -> tuple[Any, Callable[[], None]]:
        """Upload a video file to Gemini and build a file-based video part.

        Args:
            video_path: Path to the MP4 file to upload.
            fps: Playback frame rate to attach as video metadata.

        Returns:
            A tuple containing the Gemini file-data video part and a cleanup
            callback that attempts to delete the uploaded file.

        Raises:
            RuntimeError: If the upload fails or Gemini does not return a valid
            file URI.
        """
        if genai_types is None:
            raise RuntimeError(
                "google-genai is not installed. Install it to use the Gemini VLM backend."
            )

        try:
            uploaded_file = self._client.files.upload(file=video_path)
            uploaded_file = self._wait_for_uploaded_file(uploaded_file)
        except Exception as exc:
            raise RuntimeError(f"Gemini video upload failed: {exc}") from exc

        file_uri = getattr(uploaded_file, "uri", None)
        mime_type = getattr(uploaded_file, "mime_type", None) or VIDEO_MIME_TYPE
        if not isinstance(file_uri, str) or not file_uri.strip():
            raise RuntimeError("Gemini uploaded video did not return a valid file URI")

        cleanup = lambda: self._delete_uploaded_file(uploaded_file)
        return (
            genai_types.Part(
                file_data=genai_types.FileData(
                    file_uri=file_uri,
                    mime_type=mime_type,
                ),
                video_metadata=genai_types.VideoMetadata(fps=fps),
            ),
            cleanup,
        )

    def _write_temp_video(self, clip: ExtractedClip) -> str:
        """Write an extracted clip to a temporary MP4 file.

        Args:
            clip: Extracted clip to serialize as MP4.

        Returns:
            The path to the generated temporary video file.

        Raises:
            RuntimeError: If the video file cannot be created or is empty.
        """
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            video_path = tmp_file.name

        try:
            write_clip_video(
                clip,
                video_path,
                open_error_message="Failed to open a temporary MP4 writer for Gemini input",
            )
            if not Path(video_path).exists() or Path(video_path).stat().st_size <= 0:
                raise RuntimeError("Failed to encode video clip for Gemini request")
            return video_path
        except Exception:
            Path(video_path).unlink(missing_ok=True)
            raise

    def _wait_for_uploaded_file(self, uploaded_file: Any) -> Any:
        deadline = time.monotonic() + self._timeout_s
        current_file = uploaded_file
        while True:
            state_name = self._file_state_name(current_file)
            if state_name == "ACTIVE":
                return current_file
            if state_name in {"FAILED", "ERROR"}:
                raise RuntimeError(f"Gemini uploaded video entered state {state_name}")
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for Gemini video processing; last state={state_name or 'UNKNOWN'}"
                )

            file_name = getattr(current_file, "name", None)
            if not isinstance(file_name, str) or not file_name.strip():
                raise RuntimeError("Gemini uploaded video is missing a file name for polling")

            time.sleep(0.5)
            current_file = self._client.files.get(name=file_name)

    def _file_state_name(self, uploaded_file: Any) -> str:
        state = getattr(uploaded_file, "state", None)
        raw_name = getattr(state, "name", state)
        if raw_name is None:
            return ""
        return str(raw_name).strip().upper()

    def _delete_uploaded_file(self, uploaded_file: Any) -> None:
        delete_method = getattr(self._client.files, "delete", None)
        if not callable(delete_method):
            return
        file_name = getattr(uploaded_file, "name", None)
        if not isinstance(file_name, str) or not file_name.strip():
            return
        try:
            delete_method(name=file_name)
        except TypeError:
            try:
                delete_method(file_name)
            except Exception:
                return
        except Exception:
            return

    def _generate_content(self, payload: list[Any]) -> Any:
        try:
            return self._client.models.generate_content(
                model=self._model,
                contents=payload,
                config=GEMINI_GENERATION_CONFIG,
            )
        except Exception as exc:
            raise RuntimeError(f"Gemini API request failed: {exc}") from exc

    def close(self) -> None:
        """Close the underlying Gemini client when the SDK exposes close()."""
        close_method = getattr(self._client, "close", None)
        if callable(close_method):
            close_method()


def _no_cleanup_required() -> None:
    """Cleanup callback used when no cleanup action is required."""
    return None
