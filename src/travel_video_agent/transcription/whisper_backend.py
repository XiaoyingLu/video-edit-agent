"""Local speech-to-text adapter powered by faster-whisper."""

import gc
from pathlib import Path
from typing import Any

from ..resource_manager import (
    ResourceError,
    ensure_gpu_memory_available,
    ensure_memory_available,
    required_whisper_memory_gb,
    required_whisper_vram_gb,
)


class TranscriptionError(RuntimeError):
    """Raised when the local transcription backend cannot process media."""


class FasterWhisperTranscriber:
    """Run faster-whisper locally; model files are cached on disk between runs."""

    def __init__(
        self,
        model: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: Path | None = None,
    ) -> None:
        try:
            ensure_memory_available(
                required_whisper_memory_gb(model), f"faster-whisper model '{model}'"
            )
            if device.lower() == "cuda":
                ensure_gpu_memory_available(
                    required_whisper_vram_gb(model), f"faster-whisper model '{model}'"
                )
        except ResourceError as exc:
            raise TranscriptionError(str(exc)) from exc

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionError(
                "Speech transcription requires the optional dependency. "
                'Install it with: python -m pip install -e ".[transcription]"'
            ) from exc

        try:
            self._model = WhisperModel(
                model,
                device=device,
                compute_type=compute_type,
                download_root=str(download_root) if download_root else None,
            )
        except Exception as exc:
            raise TranscriptionError(f"Could not load Whisper model '{model}': {exc}") from exc

    def close(self) -> None:
        """Release CTranslate2 model references and collect freed host/GPU memory."""
        self._model = None
        gc.collect()

    def transcribe(self, media_path: Path) -> dict[str, Any]:
        """Transcribe speech in a video file and return normalized segment data."""
        try:
            segments, info = self._model.transcribe(
                str(media_path),
                beam_size=5,
                vad_filter=True,
            )
            normalized = [
                {
                    "start_seconds": max(0.0, float(segment.start)),
                    "end_seconds": max(0.0, float(segment.end)),
                    "text": segment.text.strip(),
                    "average_log_probability": getattr(segment, "avg_logprob", None),
                    "no_speech_probability": getattr(segment, "no_speech_prob", None),
                }
                for segment in segments
                if segment.text.strip() and float(segment.end) > float(segment.start)
            ]
        except Exception as exc:
            raise TranscriptionError(f"Could not transcribe {media_path}: {exc}") from exc

        return {
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
            "segments": normalized,
        }
