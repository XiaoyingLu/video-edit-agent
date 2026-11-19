"""Representative frame extraction through FFmpeg."""

import json
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
    source_stat = video_path.stat()
    source_fingerprint = {
        "size": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
    }
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
        timestamp_file = destination.with_suffix(".timestamp")
        if destination.is_file() and destination.stat().st_size > 0 and timestamp_file.is_file():
            try:
                cached = json.loads(timestamp_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached = {}
            if (
                isinstance(cached, dict)
                and cached.get("timestamp_seconds") is not None
                and abs(cached["timestamp_seconds"] - timestamp) < 0.001
                and cached.get("source") == source_fingerprint
            ):
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
            timestamp_file.write_text(
                json.dumps(
                    {"timestamp_seconds": timestamp, "source": source_fingerprint}
                )
                + "\n",
                encoding="utf-8",
            )
            extracted.append((timestamp, destination))
    return extracted
