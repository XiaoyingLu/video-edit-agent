"""Video discovery and ffprobe metadata extraction."""

import json
import shutil
from pathlib import Path
from typing import Any

from .media_runner import MediaError, run_ffmpeg
from .models import MediaInfo, ProjectAnalysis, relative_media_path

SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


class MediaToolError(MediaError):
    """Raised when ffprobe is unavailable or cannot inspect a media file."""


def discover_videos(folder: Path) -> list[Path]:
    """Recursively find supported video files in deterministic order."""
    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {folder}")

    return sorted(
        (
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES
        ),
        key=lambda path: path.as_posix().lower(),
    )


def _parse_frame_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    try:
        numerator, denominator = value.split("/", maxsplit=1)
        denominator_value = float(denominator)
        if denominator_value == 0:
            return None
        return float(numerator) / denominator_value
    except (ValueError, ZeroDivisionError):
        return None


def probe_video(path: Path, project_root: Path) -> MediaInfo:
    """Read media metadata with ffprobe; never writes to the input file."""
    executable = shutil.which("ffprobe")
    if executable is None:
        raise MediaToolError(
            "ffprobe was not found. Install FFmpeg and ensure ffprobe is available on PATH."
        )

    command = [
        executable,
        "-v",
        "error",
        "-show_entries",
        "format=duration:format_tags=creation_time:stream=codec_type,codec_name,width,height,avg_frame_rate:stream_tags=creation_time",
        "-of",
        "json",
        str(path),
    ]
    result = run_ffmpeg(
        command,
        timeout=60,
        label=f"ffprobe while inspecting {path}",
        error_type=MediaToolError,
    )

    try:
        payload: dict[str, Any] = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaToolError(f"ffprobe returned invalid JSON for {path}") from exc

    streams = payload.get("streams", [])
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    format_info = payload.get("format", {})
    duration = float(format_info.get("duration", 0.0) or 0.0)
    capture_timestamp = format_info.get("tags", {}).get("creation_time")
    if capture_timestamp is None and video_stream is not None:
        capture_timestamp = video_stream.get("tags", {}).get("creation_time")

    return MediaInfo(
        path=relative_media_path(path, project_root),
        duration_seconds=duration,
        capture_timestamp=capture_timestamp,
        width=video_stream.get("width") if video_stream else None,
        height=video_stream.get("height") if video_stream else None,
        frame_rate=_parse_frame_rate(video_stream.get("avg_frame_rate")) if video_stream else None,
        video_codec=video_stream.get("codec_name") if video_stream else None,
        audio_codec=audio_stream.get("codec_name") if audio_stream else None,
        has_video=video_stream is not None,
        has_audio=audio_stream is not None,
    )


def analyze_folder(folder: Path) -> ProjectAnalysis:
    """Analyze all supported videos under a project folder."""
    root = folder.expanduser().resolve()
    paths = discover_videos(root)
    videos = [probe_video(path, root) for path in paths]
    return ProjectAnalysis(
        project_path=str(root),
        video_count=len(videos),
        total_duration_seconds=sum(video.duration_seconds for video in videos),
        videos=videos,
    )
