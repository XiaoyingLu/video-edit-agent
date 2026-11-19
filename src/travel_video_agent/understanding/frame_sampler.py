"""Representative frame extraction through FFmpeg."""

import shutil
import subprocess
from pathlib import Path

from ..media import MediaToolError


def extract_representative_frames(
    video_path: Path,
    start_seconds: float,
    end_seconds: float,
    output_dir: Path,
    samples_per_scene: int = 3,
) -> list[tuple[float, Path]]:
    """Extract evenly spaced JPEG frames without modifying the source video."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise MediaToolError(
            "ffmpeg was not found. Install FFmpeg and ensure it is available on PATH."
        )
    if samples_per_scene < 1:
        raise ValueError("samples_per_scene must be at least 1")
    if end_seconds <= start_seconds:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    duration = end_seconds - start_seconds
    fractions = (
        [0.5]
        if samples_per_scene == 1
        else [(index + 1) / (samples_per_scene + 1) for index in range(samples_per_scene)]
    )
    extracted: list[tuple[float, Path]] = []
    for index, fraction in enumerate(fractions, start=1):
        timestamp = start_seconds + duration * fraction
        destination = output_dir / f"frame-{index:02d}.jpg"
        if destination.is_file() and destination.stat().st_size > 0:
            extracted.append((timestamp, destination))
            continue
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            "scale=1024:-2:force_original_aspect_ratio=decrease",
            "-q:v",
            "3",
            "-y",
            str(destination),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired as exc:
            raise MediaToolError(f"Frame extraction timed out for {video_path}") from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode(errors="replace").strip()
            raise MediaToolError(
                f"Could not extract frame at {timestamp:.2f}s from {video_path}: {detail}"
            ) from exc
        if destination.exists() and destination.stat().st_size:
            extracted.append((timestamp, destination))
    return extracted
