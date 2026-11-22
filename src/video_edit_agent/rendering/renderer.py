"""Render a reviewed project timeline with FFmpeg; never overwrite source media."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..artifact_tracking import register_artifact, validate_artifact
from ..editing.schemas import ProjectTimeline, VideoTimeline
from .audio_mix import append_music_input, append_music_mix, resolve_selected_music


class RenderError(RuntimeError):
    """Raised when a reviewed timeline cannot safely be rendered or verified."""


class RenderReport(BaseModel):
    """Metadata for a successfully rendered and verified MP4."""

    model_config = ConfigDict(extra="forbid")

    output_path: str
    source_timeline: str
    source_video_count: int = Field(ge=0)
    segment_count: int = Field(ge=1)
    expected_duration_seconds: float = Field(gt=0)
    actual_duration_seconds: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    video_codec: str
    audio_codec: str
    rendering_performed: bool = True


def _run_probe(executable: str, path: Path, timeout: float = 60) -> dict[str, Any]:
    command = [
        executable,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,codec_name,width,height,sample_rate,channels,pix_fmt,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise RenderError(f"ffprobe timed out while inspecting {path}") from exc
    if result.returncode != 0:
        raise RenderError(f"ffprobe could not inspect {path}: {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RenderError(f"ffprobe returned invalid JSON for {path}") from exc


def _validate_filter_graph(command: list[str], timeout: float = 120) -> None:
    """Run the exact filter graph briefly to catch parser and stream-negotiation errors."""
    try:
        encoder_index = command.index("-c:v")
    except ValueError as exc:
        raise RenderError("could not locate encoder options in generated FFmpeg command") from exc
    validation_command = [
        command[0],
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        *command[1:encoder_index],
        "-t",
        "0.5",
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            validation_command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RenderError("FFmpeg filter-graph preflight exceeded its timeout") from exc
    if result.returncode != 0:
        detail = result.stderr.strip()[-4000:] or "unknown FFmpeg filter-graph error"
        raise RenderError(f"FFmpeg filter-graph preflight failed: {detail}")


def _validate_source_stream_metadata(payload: dict[str, Any], path: Path) -> None:
    """Reject unusable video dimensions or malformed audio stream metadata."""
    streams = payload.get("streams", [])
    video_streams = [item for item in streams if item.get("codec_type") == "video"]
    if not video_streams:
        raise RenderError(f"Timeline source has no video stream: {path}")
    for stream in video_streams:
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
        if width < 2 or height < 2:
            raise RenderError(f"Timeline source has invalid video dimensions: {path}")
    for stream in streams:
        if stream.get("codec_type") != "audio":
            continue
        sample_rate = stream.get("sample_rate")
        channels = stream.get("channels")
        if sample_rate is not None and int(sample_rate) <= 0:
            raise RenderError(f"Timeline source has an invalid audio sample rate: {path}")
        if channels is not None and int(channels) <= 0:
            raise RenderError(f"Timeline source has an invalid audio channel count: {path}")


def _probe_source(
    ffprobe: str, root: Path, video: VideoTimeline
) -> tuple[Path, dict[str, Any], float]:
    relative = Path(video.path)
    if relative.is_absolute():
        raise RenderError(f"Timeline source path must be project-relative: {video.path}")
    source = (root / relative).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise RenderError(f"Timeline source escapes the project folder: {video.path}") from exc
    if not source.is_file():
        raise RenderError(f"Timeline source file does not exist: {source}")

    payload = _run_probe(ffprobe, source)
    streams = payload.get("streams", [])
    video_stream = next((item for item in streams if item.get("codec_type") == "video"), None)
    if video_stream is None:
        raise RenderError(f"Timeline source has no video stream: {source}")
    _validate_source_stream_metadata(payload, source)
    duration = float(payload.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise RenderError(f"Timeline source has an invalid duration: {source}")
    if abs(duration - video.source_duration_seconds) > max(0.25, duration * 0.005):
        raise RenderError(
            f"Source duration changed since timeline creation for {video.path}: "
            f"timeline={video.source_duration_seconds:.3f}s, actual={duration:.3f}s"
        )
    return source, payload, duration


def _seconds(value: float) -> str:
    return f"{value:.6f}"


def _subtitle_force_style(
    style_name: str, font_size: int, position: str, width: int, height: int
) -> str:
    """Build a fixed ASS subtitle preset with margins scaled to output dimensions."""
    if style_name not in {"classic", "cinematic", "boxed", "minimal"}:
        raise ValueError("subtitle_style must be classic, cinematic, boxed, or minimal")
    if position not in {"bottom", "center", "top"}:
        raise ValueError("subtitle_position must be bottom, center, or top")
    if not 12 <= font_size <= 72:
        raise ValueError("subtitle_font_size must be between 12 and 72")
    if width < 2 or height < 2:
        raise ValueError("subtitle output dimensions must be at least 2 pixels")

    alignment = {"bottom": 2, "center": 5, "top": 8}[position]
    if style_name == "classic":
        margin = 0 if position == "center" else 48
        return (
            f"FontName=DejaVu Sans,FontSize={font_size},"
            f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            f"BorderStyle=1,Outline=2,Shadow=1,Alignment={alignment},MarginV={margin}"
        )

    responsive_size = max(12, min(72, round(font_size * height / 1080)))
    if style_name == "cinematic":
        margin = max(32, round(height * 0.06))
        appearance = (
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H00101010,"
            "BorderStyle=1,Outline=3,Shadow=1,Bold=1"
        )
    elif style_name == "boxed":
        margin = max(32, round(height * 0.06))
        appearance = (
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,"
            "BackColour=&H80000000,BorderStyle=3,Outline=12,Shadow=0"
        )
    else:
        margin = max(24, round(height * 0.05))
        appearance = (
            "PrimaryColour=&H00FFFFFF,OutlineColour=&H66000000,"
            "BorderStyle=1,Outline=1,Shadow=1,Bold=0"
        )
    return (
        f"FontName=DejaVu Sans,FontSize={responsive_size},"
        f"{appearance},Alignment={alignment},MarginV={margin}"
    )


def _build_ffmpeg_command(
    ffmpeg: str,
    sources: list[tuple[Path, dict[str, Any], float]],
    segments: list[tuple[int, float, float, bool]],
    output: Path,
    width: int,
    height: int,
    frame_rate: float,
    subtitle_path: Path | None = None,
    subtitle_font_size: int = 28,
    subtitle_position: str = "bottom",
    subtitle_style: str = "classic",
    music_path: Path | None = None,
    music_mode: str = "blend",
    music_volume: float = 0.22,
    music_fade_in: float = 1.0,
    music_fade_out: float = 1.5,
    music_ducking: bool = False,
) -> list[str]:
    command = [ffmpeg, "-hide_banner", "-nostdin", "-y"]
    for source_index, _, _ in sources:
        command.extend(["-i", str(source_index)])
    music_input_index = append_music_input(command, music_path)

    filters: list[str] = []
    concat_inputs: list[str] = []
    for index, (input_index, start, end, has_audio) in enumerate(segments):
        duration = end - start
        video_label = f"v{index}"
        audio_label = f"a{index}"
        video_filter = (
            f"[{input_index}:v:0]trim=start={_seconds(start)}:end={_seconds(end)},"
            f"setpts=PTS-STARTPTS,"
            f"scale=w={width}:h={height}:force_original_aspect_ratio=decrease,"
            f"pad=width={width}:height={height}:x=(ow-iw)/2:y=(oh-ih)/2,"
            f"setsar=1,fps={_seconds(frame_rate)},format=yuv420p[{video_label}]"
        )
        filters.append(video_filter)
        if has_audio:
            filters.append(
                f"[{input_index}:a:0]atrim=start={_seconds(start)}:end={_seconds(end)},"
                f"asetpts=PTS-STARTPTS,aresample=48000,"
                f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
                f"[{audio_label}]"
            )
        else:
            filters.append(
                f"anullsrc=r=48000:cl=stereo,atrim=duration={_seconds(duration)}[{audio_label}]"
            )
        concat_inputs.extend([f"[{video_label}]", f"[{audio_label}]"])

    filters.append(f"{''.join(concat_inputs)}concat=n={len(segments)}:v=1:a=1[outv][outa]")
    audio_output = append_music_mix(
        filters,
        base_label="outa",
        music_input_index=music_input_index,
        duration_seconds=sum(end - start for _, start, end, _ in segments),
        mode=music_mode,
        volume=music_volume,
        fade_in_seconds=music_fade_in,
        fade_out_seconds=music_fade_out,
        ducking=music_ducking,
    )
    video_output = "[outv]"
    if subtitle_path is not None:
        escaped_path = (
            str(subtitle_path.resolve())
            .replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("'", "\\'")
        )
        style = _subtitle_force_style(
            subtitle_style, subtitle_font_size, subtitle_position, width, height
        )
        filters.append(
            f"[outv]subtitles=filename='{escaped_path}':force_style='{style}'[subtitled]"
        )
        video_output = "[subtitled]"
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            video_output,
            "-map",
            f"[{audio_output}]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    return command


def render_project(
    folder: Path,
    timeline_path: Path | None = None,
    output_path: Path | None = None,
    *,
    overwrite: bool = False,
    timeout_seconds: float = 3600,
    subtitle_path: Path | None = None,
    subtitle_font_size: int = 28,
    subtitle_position: str = "bottom",
    subtitle_style: str = "classic",
    music_path: Path | None = None,
    music_mode: str = "blend",
    music_volume: float = 0.22,
    music_fade_in: float = 1.0,
    music_fade_out: float = 1.5,
    music_ducking: bool = False,
    dry_run: bool = False,
) -> RenderReport | None:
    """Render the approved retained intervals into one MP4 and verify the result.

    All source paths are constrained to the project root. Rendering uses a temporary
    output followed by an atomic replace, and never writes to a source file.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    if not 12 <= subtitle_font_size <= 72:
        raise ValueError("subtitle_font_size must be between 12 and 72")
    if subtitle_position not in {"bottom", "center", "top"}:
        raise ValueError("subtitle_position must be bottom, center, or top")
    if subtitle_style not in {"classic", "cinematic", "boxed", "minimal"}:
        raise ValueError("subtitle_style must be classic, cinematic, boxed, or minimal")
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project folder does not exist: {root}")
    selected_music = (
        None if music_mode == "original" else resolve_selected_music(root, music_path=music_path)
    )

    timeline_file = (
        (timeline_path or root / ".video-agent" / "timeline.json").expanduser().resolve()
    )
    if not timeline_file.is_file():
        raise FileNotFoundError(f"Missing {timeline_file}; run 'video-agent build-timeline' first.")
    validate_artifact(root, timeline_file)
    timeline = ProjectTimeline.model_validate_json(timeline_file.read_text(encoding="utf-8"))
    if Path(timeline.project_path).resolve() != root:
        raise RenderError("timeline belongs to a different project folder")
    if timeline.rendering_performed:
        raise RenderError("timeline artifact must describe an edit decision, not a prior render")

    destination = (output_path or root / ".video-agent" / "rendered.mp4").expanduser().resolve()
    if destination.suffix.lower() != ".mp4":
        raise RenderError("render output must use the .mp4 extension")
    subtitle_file = subtitle_path.expanduser().resolve() if subtitle_path else None
    if subtitle_file is not None and not subtitle_file.is_file():
        raise FileNotFoundError(
            f"Subtitle file does not exist: {subtitle_file}; run 'video-agent subtitles' first."
        )
    if subtitle_file is not None:
        validate_artifact(root, subtitle_file)
    if subtitle_file is not None and destination == subtitle_file:
        raise RenderError("render output cannot overwrite the subtitle file")
    if destination == timeline_file:
        raise RenderError("render output cannot overwrite the timeline artifact")
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)

    if not timeline.videos:
        raise RenderError("timeline contains no source videos")
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RenderError("ffprobe was not found; install FFmpeg and ensure it is on PATH")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RenderError("ffmpeg was not found; install FFmpeg and ensure it is on PATH")
    source_entries = [_probe_source(ffprobe, root, video) for video in timeline.videos]

    source_paths = {entry[0] for entry in source_entries}
    if destination in source_paths:
        raise RenderError("render output cannot overwrite a source video")
    if selected_music is not None and destination == selected_music:
        raise RenderError("render output cannot overwrite the selected soundtrack")
    if destination.exists() and not overwrite and not dry_run:
        raise FileExistsError(
            f"Render output already exists: {destination}; pass --overwrite to replace it."
        )

    segments: list[tuple[int, float, float, bool]] = []
    for source_index, (video, (_, payload, duration)) in enumerate(
        zip(timeline.videos, source_entries, strict=True)
    ):
        streams = payload.get("streams", [])
        has_audio = any(item.get("codec_type") == "audio" for item in streams)
        for segment in video.segments:
            if segment.start_seconds < 0 or segment.end_seconds > duration + 0.01:
                raise RenderError(f"timeline segment is outside source bounds: {video.path}")
            if segment.end_seconds <= segment.start_seconds:
                raise RenderError(f"timeline contains an empty segment: {video.path}")
            segments.append((source_index, segment.start_seconds, segment.end_seconds, has_audio))
    if not segments:
        raise RenderError("timeline retains no footage; refusing to create an empty video")

    first_video_stream = next(
        item for item in source_entries[0][1]["streams"] if item.get("codec_type") == "video"
    )
    width = int(first_video_stream.get("width") or 0)
    height = int(first_video_stream.get("height") or 0)
    width -= width % 2
    height -= height % 2
    if width < 2 or height < 2:
        raise RenderError("could not determine a valid output resolution from the first source")
    frame_rate = 30.0

    expected_duration = sum(end - start for _, start, end, _ in segments)
    temporary_path = destination
    if not dry_run:
        temporary_file = tempfile.NamedTemporaryFile(
            prefix=".video-agent-render-", suffix=".mp4", dir=destination.parent, delete=False
        )
        temporary_path = Path(temporary_file.name)
        temporary_file.close()
    try:
        command = _build_ffmpeg_command(
            ffmpeg,
            [(entry[0], entry[1], entry[2]) for entry in source_entries],
            segments,
            temporary_path,
            width,
            height,
            frame_rate,
            subtitle_path=subtitle_file,
            subtitle_font_size=subtitle_font_size,
            subtitle_position=subtitle_position,
            subtitle_style=subtitle_style,
            music_path=selected_music,
            music_mode=music_mode,
            music_volume=music_volume,
            music_fade_in=music_fade_in,
            music_fade_out=music_fade_out,
            music_ducking=music_ducking,
        )
        if dry_run:
            _validate_filter_graph(command, timeout=min(timeout_seconds, 120))
            return None
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RenderError(f"FFmpeg render exceeded the {timeout_seconds:.0f}s timeout") from exc
        if result.returncode != 0:
            detail = result.stderr.strip()[-4000:] or "unknown FFmpeg error"
            raise RenderError(f"FFmpeg render failed: {detail}")
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise RenderError("FFmpeg completed without producing a non-empty MP4")

        output_probe = _run_probe(ffprobe, temporary_path)
        output_streams = output_probe.get("streams", [])
        output_video = next(
            (item for item in output_streams if item.get("codec_type") == "video"), None
        )
        output_audio = next(
            (item for item in output_streams if item.get("codec_type") == "audio"), None
        )
        actual_duration = float(output_probe.get("format", {}).get("duration") or 0)
        if output_video is None or output_audio is None or actual_duration <= 0:
            raise RenderError("rendered MP4 is missing a valid video or audio stream")
        if output_video.get("codec_name") != "h264" or output_audio.get("codec_name") != "aac":
            raise RenderError(
                "rendered MP4 does not contain the expected H.264 video and AAC audio"
            )
        duration_tolerance = max(0.75, len(segments) / frame_rate * 2)
        if abs(actual_duration - expected_duration) > duration_tolerance:
            raise RenderError(
                f"rendered duration mismatch: expected about {expected_duration:.2f}s, "
                f"got {actual_duration:.2f}s"
            )

        temporary_path.replace(destination)
    except Exception:
        if not dry_run:
            temporary_path.unlink(missing_ok=True)
        raise

    dependencies = [timeline_file]
    if subtitle_file is not None:
        dependencies.append(subtitle_file)
    if selected_music is not None:
        dependencies.append(selected_music)
    register_artifact(root, destination, dependencies=dependencies, verify_content=False)

    return RenderReport(
        output_path=str(destination),
        source_timeline=str(timeline_file),
        source_video_count=len(timeline.videos),
        segment_count=len(segments),
        expected_duration_seconds=expected_duration,
        actual_duration_seconds=actual_duration,
        width=int(output_video.get("width") or width),
        height=int(output_video.get("height") or height),
        video_codec=str(output_video.get("codec_name") or "unknown"),
        audio_codec=str(output_audio.get("codec_name") or "unknown"),
    )
