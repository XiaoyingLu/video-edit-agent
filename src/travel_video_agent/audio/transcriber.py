"""Local faster-whisper adapter with temporary FFmpeg audio extraction."""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterator


class TranscriptionError(RuntimeError):
    """Raised when local audio extraction or transcription fails."""


class FasterWhisperTranscriber:
    """Transcribe speech locally; no audio is uploaded to a remote service."""

    def __init__(
        self,
        model: str = "tiny",
        language: str | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionError(
                "Audio transcription dependencies are missing. Rebuild the Docker image "
                "or install the project with the 'audio' extra."
            ) from exc

        self.model_name = model
        self.language = language
        try:
            self.model = WhisperModel(model, device=device, compute_type=compute_type)
        except Exception as exc:
            raise TranscriptionError(f"Could not load faster-whisper model '{model}': {exc}") from exc

    def transcribe(
        self, video_path: Path, cache_dir: Path, vad_filter: bool = True
    ) -> tuple[list[dict[str, Any]], str | None, float | None]:
        """Extract mono 16 kHz PCM temporarily and return normalized speech segments."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise TranscriptionError(
                "ffmpeg was not found. Install FFmpeg and ensure it is available on PATH."
            )

        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="transcribe-", suffix=".wav", dir=cache_dir, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)

            command = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(temporary_path),
            ]
            try:
                subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
            except subprocess.TimeoutExpired as exc:
                raise TranscriptionError(f"Audio extraction timed out for {video_path}") from exc
            except subprocess.CalledProcessError as exc:
                detail = exc.stderr.strip() or "unknown FFmpeg error"
                raise TranscriptionError(f"Could not extract audio from {video_path}: {detail}") from exc

            try:
                raw_segments, info = self.model.transcribe(
                    str(temporary_path),
                    language=self.language,
                    vad_filter=vad_filter,
                    word_timestamps=False,
                )
                segments = [
                    {
                        "start_seconds": max(0.0, float(item.start)),
                        "end_seconds": max(0.0, float(item.end)),
                        "text": str(item.text).strip(),
                        "confidence": _segment_confidence(item),
                    }
                    for item in raw_segments
                    if float(item.end) > float(item.start) and str(item.text).strip()
                ]
            except Exception as exc:
                raise TranscriptionError(f"Transcription failed for {video_path}: {exc}") from exc

            return segments, getattr(info, "language", None), _probability(
                getattr(info, "language_probability", None)
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def _probability(value: Any) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _segment_confidence(segment: Any) -> float:
    """Convert Whisper's average log probability to a bounded confidence heuristic."""
    value = getattr(segment, "avg_logprob", None)
    if value is None:
        return 0.5
    import math

    return max(0.0, min(1.0, math.exp(float(value))))
