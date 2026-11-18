"""Render an explicitly approved shot timeline with real FFmpeg transitions."""

import shutil
import subprocess
import tempfile
from pathlib import Path

from ..artifact_tracking import register_artifact, validate_artifact
from ..editing.planning_schemas import VideoProjectPlan
from ..editing.timeline_schemas import ProjectDraftTimeline
from .renderer import (
    RenderError,
    RenderReport,
    _run_probe,
    _seconds,
    _validate_filter_graph,
    _validate_source_stream_metadata,
)


def _safe_source(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise RenderError(f"Timeline source path must be project-relative: {relative_path}")
    source = (root / relative).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise RenderError(f"Timeline source escapes the project folder: {relative_path}") from exc
    if not source.is_file():
        raise RenderError(f"Timeline source file does not exist: {source}")
    return source


def _resolve_output_dimensions(
    root: Path, fallback_width: int, fallback_height: int
) -> tuple[int, int]:
    """Resolve the requested output profile from the current creative project plan."""
    plan_file = root / ".video-agent" / "video-project-plan.json"
    if not plan_file.is_file():
        return fallback_width, fallback_height
    validate_artifact(root, plan_file)
    plan = VideoProjectPlan.model_validate_json(plan_file.read_text(encoding="utf-8"))
    if Path(plan.project_path).resolve() != root:
        raise RenderError("video project plan belongs to a different project folder")
    long_side = {"720p": 1280, "1080p": 1920, "4k": 3840}[plan.brief.resolution]
    short_side = {"720p": 720, "1080p": 1080, "4k": 2160}[plan.brief.resolution]
    if plan.brief.aspect_ratio == "16:9":
        return long_side, short_side
    if plan.brief.aspect_ratio == "9:16":
        return short_side, long_side
    return short_side, short_side


def _build_shot_render_command(
    ffmpeg: str,
    clips: list[tuple[Path, float, float, bool, str, float]],
    output: Path,
    width: int,
    height: int,
    frame_rate: float = 30.0,
    fit_mode: str = "crop",
) -> list[str]:
    """Build a filter graph that preserves shot order and applies approved transitions."""
    if fit_mode not in {"crop", "pad"}:
        raise ValueError("fit_mode must be crop or pad")
    command = [ffmpeg, "-hide_banner", "-nostdin", "-y"]
    for source, *_ in clips:
        command.extend(["-i", str(source)])

    filters: list[str] = []
    for index, (_, start, end, has_audio, _, _) in enumerate(clips):
        duration = end - start
        fit_filters = (
            f"scale=w={width}:h={height}:force_original_aspect_ratio=increase,crop={width}:{height}"
            if fit_mode == "crop"
            else f"scale=w={width}:h={height}:force_original_aspect_ratio=decrease,"
            f"pad=width={width}:height={height}:x=(ow-iw)/2:y=(oh-ih)/2"
        )
        filters.append(
            f"[{index}:v:0]trim=start={_seconds(start)}:end={_seconds(end)},"
            f"setpts=PTS-STARTPTS,{fit_filters},"
            f"setsar=1,fps={_seconds(frame_rate)},settb=AVTB,"
            f"format=yuv420p[v{index}]"
        )
        if has_audio:
            filters.append(
                f"[{index}:a:0]atrim=start={_seconds(start)}:end={_seconds(end)},"
                f"asetpts=PTS-STARTPTS,aresample=48000,"
                f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a{index}]"
            )
        else:
            filters.append(
                f"anullsrc=r=48000:cl=stereo,atrim=duration={_seconds(duration)},"
                f"asetpts=PTS-STARTPTS,aresample=48000,"
                f"aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a{index}]"
            )

    current_video = "v0"
    current_audio = "a0"
    current_duration = clips[0][2] - clips[0][1]
    for index in range(1, len(clips)):
        _, start, end, _, transition, transition_duration = clips[index]
        clip_duration = end - start
        next_video = f"v{index}"
        next_audio = f"a{index}"
        output_video = f"vx{index}"
        output_audio = f"ax{index}"

        if transition == "cut" or transition_duration <= 0:
            filters.append(
                f"[{current_video}][{current_audio}][{next_video}][{next_audio}]"
                f"concat=n=2:v=1:a=1[{output_video}][{output_audio}]"
            )
            current_duration += clip_duration
        else:
            if transition_duration > min(current_duration, clip_duration) * 0.25 + 0.001:
                raise RenderError(
                    f"Transition before clip {index + 1} is too long for adjacent shots"
                )
            transition_name = "fade" if transition == "crossfade" else "dissolve"
            offset = current_duration - transition_duration
            filters.append(
                f"[{current_video}][{next_video}]xfade=transition={transition_name}:"
                f"duration={_seconds(transition_duration)}:offset={_seconds(offset)}"
                f"[{output_video}]"
            )
            filters.append(
                f"[{current_audio}][{next_audio}]"
                f"acrossfade=d={_seconds(transition_duration)}:c1=tri:c2=tri"
                f"[{output_audio}]"
            )
            current_duration += clip_duration - transition_duration

        current_video = output_video
        current_audio = output_audio

    filters.append(f"[{current_video}]format=yuv420p[vfinal]")
    filters.append(f"[{current_audio}]aresample=48000[audiofinal]")
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[vfinal]",
            "-map",
            "[audiofinal]",
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


def render_approved_shot_timeline(
    folder: Path,
    timeline_path: Path | None = None,
    output_path: Path | None = None,
    *,
    overwrite: bool = False,
    timeout_seconds: float = 3600,
    dry_run: bool = False,
    fit_mode: str = "crop",
) -> RenderReport | None:
    """Render only approved shots after every proposed shot has a decision."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    if fit_mode not in {"crop", "pad"}:
        raise ValueError("fit_mode must be crop or pad")
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project folder does not exist: {root}")

    timeline_file = (
        (timeline_path or root / ".video-agent" / "optimized-timeline.json").expanduser().resolve()
    )
    if not timeline_file.is_file():
        raise FileNotFoundError(
            f"Missing {timeline_file}; run 'video-agent optimize-timeline' first."
        )
    validate_artifact(root, timeline_file)
    timeline = ProjectDraftTimeline.model_validate_json(timeline_file.read_text(encoding="utf-8"))
    if Path(timeline.project_path).resolve() != root:
        raise RenderError("optimized timeline belongs to a different project folder")
    if timeline.rendering_performed:
        raise RenderError("optimized timeline must be a review artifact, not a render report")
    if not timeline.timeline_approved:
        raise RenderError(
            "shot timeline has not been fully approved; run 'video-agent review-timeline'"
        )
    if any(clip.decision == "pending" for clip in timeline.clips):
        raise RenderError("all proposed shots must have an explicit review decision")
    approved = [clip for clip in timeline.clips if clip.decision == "approved"]
    if not approved:
        raise RenderError("shot timeline contains no approved clips")

    destination = (
        (output_path or root / ".video-agent" / "rendered-draft.mp4").expanduser().resolve()
    )
    if destination.suffix.lower() != ".mp4":
        raise RenderError("render output must use the .mp4 extension")
    if destination in {timeline_file, root / ".video-agent" / "optimized-timeline.json"}:
        raise RenderError("render output cannot overwrite the optimized timeline")
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite and not dry_run:
        raise FileExistsError(
            f"Render output already exists: {destination}; pass --overwrite to replace it."
        )

    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if ffprobe is None or ffmpeg is None:
        raise RenderError("ffmpeg and ffprobe must be installed and available on PATH")

    source_cache: dict[Path, tuple[dict, float]] = {}
    clip_inputs: list[tuple[Path, float, float, bool, str, float]] = []
    source_paths: set[Path] = set()
    previous_sequence_index: int | None = None
    for clip in approved:
        source = _safe_source(root, clip.video_path)
        source_paths.add(source)
        if source not in source_cache:
            payload = _run_probe(ffprobe, source)
            streams = payload.get("streams", [])
            video_stream = next(
                (item for item in streams if item.get("codec_type") == "video"), None
            )
            if video_stream is None:
                raise RenderError(f"Timeline source has no video stream: {source}")
            _validate_source_stream_metadata(payload, source)
            duration = float(payload.get("format", {}).get("duration") or 0)
            if duration <= 0:
                raise RenderError(f"Timeline source has an invalid duration: {source}")
            source_cache[source] = (payload, duration)

        payload, source_duration = source_cache[source]
        if clip.source_end_seconds > source_duration + 0.01:
            raise RenderError(f"Approved clip exceeds source duration: {clip.clip_id}")
        if clip.source_start_seconds < 0 or clip.source_end_seconds <= clip.source_start_seconds:
            raise RenderError(f"Approved clip has invalid source bounds: {clip.clip_id}")
        streams = payload.get("streams", [])
        has_audio = any(item.get("codec_type") == "audio" for item in streams)
        # The first approved shot always starts with a hard cut, even if earlier
        # proposed shots carrying its transition metadata were rejected.
        adjacent_to_previous = (
            previous_sequence_index is not None
            and clip.sequence_index == previous_sequence_index + 1
        )
        transition = clip.transition_in if adjacent_to_previous else "cut"
        transition_duration = clip.transition_duration_seconds if adjacent_to_previous else 0.0
        clip_inputs.append(
            (
                source,
                clip.source_start_seconds,
                clip.source_end_seconds,
                has_audio,
                transition,
                transition_duration,
            )
        )
        previous_sequence_index = clip.sequence_index

    if destination in source_paths:
        raise RenderError("render output cannot overwrite a source video")
    first_payload = source_cache[clip_inputs[0][0]][0]
    first_video = next(
        item for item in first_payload["streams"] if item.get("codec_type") == "video"
    )
    width = int(first_video.get("width") or 0)
    height = int(first_video.get("height") or 0)
    width -= width % 2
    height -= height % 2
    if width < 2 or height < 2:
        raise RenderError("could not determine a valid output resolution from the first source")
    width, height = _resolve_output_dimensions(root, width, height)

    expected_duration = sum(end - start for _, start, end, *_ in clip_inputs) - sum(
        transition_duration
        for *_, transition, transition_duration in clip_inputs[1:]
        if transition != "cut"
    )
    if expected_duration <= 0:
        raise RenderError("approved shot sequence has no positive output duration")
    temporary_path = destination
    if not dry_run:
        temporary_file = tempfile.NamedTemporaryFile(
            prefix=".video-agent-shot-render-", suffix=".mp4", dir=destination.parent, delete=False
        )
        temporary_path = Path(temporary_file.name)
        temporary_file.close()
    try:
        command = _build_shot_render_command(
            ffmpeg, clip_inputs, temporary_path, width, height, fit_mode=fit_mode
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
            raise RenderError(f"FFmpeg shot render failed: {detail}")
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
            raise RenderError("rendered MP4 does not contain the expected H.264/AAC streams")
        actual_width = int(output_video.get("width") or 0)
        actual_height = int(output_video.get("height") or 0)
        if (actual_width, actual_height) != (width, height):
            raise RenderError(
                f"rendered dimensions mismatch: expected {width}x{height}, "
                f"got {actual_width}x{actual_height}"
            )
        tolerance = max(0.75, len(clip_inputs) / 30.0 * 2)
        if abs(actual_duration - expected_duration) > tolerance:
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
    plan_file = root / ".video-agent" / "video-project-plan.json"
    if plan_file.is_file():
        dependencies.append(plan_file)
    register_artifact(root, destination, dependencies=dependencies, verify_content=False)

    return RenderReport(
        output_path=str(destination),
        source_timeline=str(timeline_file),
        source_video_count=len(source_paths),
        segment_count=len(clip_inputs),
        expected_duration_seconds=expected_duration,
        actual_duration_seconds=actual_duration,
        width=int(output_video.get("width") or width),
        height=int(output_video.get("height") or height),
        video_codec=str(output_video.get("codec_name") or "unknown"),
        audio_codec=str(output_audio.get("codec_name") or "unknown"),
    )
