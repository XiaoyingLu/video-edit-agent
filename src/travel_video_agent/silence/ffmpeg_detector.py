"""FFmpeg silencedetect adapter."""

import re
import shutil
import subprocess
from pathlib import Path

from .schemas import SilenceInterval

_SILENCE_START = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?).*?silence_duration:\s*(\d+(?:\.\d+)?)")


class SilenceDetectionError(RuntimeError):
    """Raised when FFmpeg is unavailable or cannot analyze an audio stream."""


class FFmpegSilenceDetector:
    """Detect low-volume intervals without changing the source media."""

    def detect(
        self,
        media_path: Path,
        duration_seconds: float,
        noise_threshold_db: float = -35.0,
        min_silence_duration: float = 1.5,
    ) -> list[SilenceInterval]:
        if not -90.0 <= noise_threshold_db <= -1.0:
            raise ValueError("noise_threshold_db must be between -90 and -1 dB")
        if min_silence_duration <= 0:
            raise ValueError("min_silence_duration must be positive")

        executable = shutil.which("ffmpeg")
        if executable is None:
            raise SilenceDetectionError(
                "ffmpeg was not found. Install FFmpeg and ensure it is available on PATH."
            )

        command = [
            executable,
            "-hide_banner",
            "-nostats",
            "-i",
            str(media_path),
            "-af",
            f"silencedetect=noise={noise_threshold_db}dB:d={min_silence_duration}",
            "-f",
            "null",
            "-",
        ]
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=3600,
            )
        except subprocess.TimeoutExpired as exc:
            raise SilenceDetectionError(
                f"FFmpeg silence detection timed out for {media_path}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or "unknown FFmpeg error"
            raise SilenceDetectionError(
                f"Could not analyze audio in {media_path}: {detail}"
            ) from exc

        return self.parse_output(result.stderr, duration_seconds)

    @staticmethod
    def parse_output(log: str, duration_seconds: float) -> list[SilenceInterval]:
        """Parse FFmpeg silencedetect events, including trailing silence."""
        intervals: list[SilenceInterval] = []
        pending_start: float | None = None
        for line in log.splitlines():
            start_match = _SILENCE_START.search(line)
            if start_match:
                pending_start = max(0.0, float(start_match.group(1)))
                continue

            end_match = _SILENCE_END.search(line)
            if end_match and pending_start is not None:
                end = min(duration_seconds, max(pending_start, float(end_match.group(1))))
                start = min(pending_start, end)
                if end > start:
                    intervals.append(
                        SilenceInterval(
                            start_seconds=start,
                            end_seconds=end,
                            duration_seconds=end - start,
                        )
                    )
                pending_start = None

        if pending_start is not None:
            start = min(pending_start, duration_seconds)
            if duration_seconds > start:
                intervals.append(
                    SilenceInterval(
                        start_seconds=start,
                        end_seconds=duration_seconds,
                        duration_seconds=duration_seconds - start,
                    )
                )
        return intervals
