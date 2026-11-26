"""Render an explicitly approved shot timeline with real FFmpeg transitions."""

import re
import shutil
import tempfile
from pathlib import Path

from ..artifact_tracking import register_artifact, validate_artifact
from ..editing.planning_schemas import VideoProjectPlan
from ..editing.timeline_schemas import ProjectDraftTimeline
from ..editing.user_overrides import apply_user_overrides
from ..media_runner import run_ffmpeg
from ..music.schemas import MusicAnalysis
from .audio_mix import append_music_input, append_music_mix, resolve_selected_music
from .face_reframing import detect_face_crop_paths
from .overlay_filters import _append_asset_overlay_filters, _append_title_overlay_filters
from .renderer import (
    RenderError,
    RenderReport,
    _run_probe,
    _seconds,
    _validate_filter_graph,
    _validate_source_stream_metadata,
)
from .social_crop_anchors import load_crop_anchors
from .title_overlays import TitleOverlay, resolve_asset_overlays


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


def _requested_output_dimensions(aspect_ratio: str, resolution: str) -> tuple[int, int]:
    """Resolve a validated aspect-ratio and resolution pair to even output dimensions."""
    if aspect_ratio not in {"16:9", "9:16", "1:1"}:
        raise ValueError("output_aspect_ratio must be 16:9, 9:16, or 1:1")
    if resolution not in {"720p", "1080p", "4k"}:
        raise ValueError("output_resolution must be 720p, 1080p, or 4k")
    long_side = {"720p": 1280, "1080p": 1920, "4k": 3840}[resolution]
    short_side = {"720p": 720, "1080p": 1080, "4k": 2160}[resolution]
    if aspect_ratio == "16:9":
        return long_side, short_side
    if aspect_ratio == "9:16":
        return short_side, long_side
    return short_side, short_side


def _crop_path_expression(path: list[tuple[float, ...]], axis: int) -> str:
    """Build a piecewise-linear FFmpeg expression for a normalized crop coordinate."""
    if len(path) == 1:
        return f"{path[0][axis]:.6f}"
    expression = f"{path[-1][axis]:.6f}"
    for index in range(len(path) - 2, -1, -1):
        current = path[index]
        following = path[index + 1]
        start, end = current[0], following[0]
        value_start, value_end = current[axis], following[axis]
        if end <= start:
            continue
        interpolation = (
            f"({value_start:.6f}+({value_end - value_start:.6f})*(t-{start:.6f})/{end - start:.6f})"
        )
        expression = f"if(lt(t,{end:.6f}),{interpolation},{expression})"
    first_time, first_value = path[0][0], path[0][axis]
    if first_time > 0:
        expression = f"if(lt(t,{first_time:.6f}),{first_value:.6f},{expression})"
    return expression


_XFADE_TRANSITIONS = {
    "crossfade": "fade",
    "dissolve": "dissolve",
    "fadeblack": "fadeblack",
    "wipeleft": "wipeleft",
    "slideleft": "slideleft",
    "circleopen": "circleopen",
}


def _build_beat_zoom_paths(
    root: Path,
    selected_music: Path | None,
    clips: list[tuple[Path, float, float, bool, str, float]],
) -> list[list[tuple[float, float, float]]] | None:
    """Create short centered zoom pulses from a validated soundtrack beat grid."""
    analysis_file = root / ".video-agent" / "music-analysis.json"
    if selected_music is None or not analysis_file.is_file():
        return None
    try:
        validate_artifact(root, analysis_file)
        analysis = MusicAnalysis.model_validate_json(analysis_file.read_text(encoding="utf-8"))
        music_reference = Path(analysis.music_path).expanduser()
        analyzed_music = (
            music_reference if music_reference.is_absolute() else root / music_reference
        ).resolve()
    except (OSError, ValueError) as exc:
        raise RenderError(f"could not load beat analysis for punch-in: {exc}") from exc
    if analyzed_music != selected_music.resolve() or analysis.beat_confidence < 0.1:
        return None
    beats = sorted(analysis.beat_times_seconds)
    if not beats:
        return None

    paths: list[list[tuple[float, float, float]]] = []
    timeline_start = 0.0
    for index, clip in enumerate(clips):
        duration = clip[2] - clip[1]
        points: dict[float, float] = {0.0: 1.0, round(duration, 4): 1.0}
        for beat in beats:
            local_beat = beat - timeline_start
            if 0 <= local_beat <= duration:
                for offset, zoom in (
                    (-0.06, 1.0),
                    (0.0, 1.06),
                    (0.08, 1.06),
                    (0.24, 1.0),
                ):
                    timestamp = round(min(duration, max(0.0, local_beat + offset)), 4)
                    points[timestamp] = zoom
        paths.append([(time, zoom, zoom) for time, zoom in sorted(points.items())])
        timeline_start += duration
        if index + 1 < len(clips) and clips[index + 1][4] != "cut":
            timeline_start -= clips[index + 1][5]
    return paths


def _beat_zoom_filter(path: list[tuple[float, float, float]], width: int, height: int) -> str:
    """Return a frame-evaluated centered zoom filter for a normalized zoom path."""
    if not path:
        return ""
    expression = _crop_path_expression(path, 1)
    return (
        f"scale=w='iw*({expression})':h='ih*({expression})':eval=frame,"
        f"crop={width}:{height}:(iw-ow)/2:(ih-oh)/2,"
    )


def _validate_xfade_transitions(
    ffmpeg: str,
    clip_inputs: list[tuple[Path, float, float, bool, str, float]],
) -> None:
    """Check the installed FFmpeg xfade enum before building a requested transition."""
    requested = {transition for *_, transition, _ in clip_inputs if transition != "cut"}
    if not requested:
        return

    result = run_ffmpeg(
        [ffmpeg, "-hide_banner", "-h", "filter=xfade"],
        timeout=30,
        label="FFmpeg xfade capability preflight",
        error_type=RenderError,
    )
    help_text = "\n".join(filter(None, (result.stdout or "", result.stderr or "")))
    supported = set(
        re.findall(
            r"^[ \t]+([a-z][a-z0-9_]*)[ \t]+-?\d+(?:[ \t]+[.A-Z]+(?:[ \t]+[^\r\n]*)?)?[ \t]*$",
            help_text,
            re.MULTILINE,
        )
    )
    unsupported = sorted(
        transition
        for transition in requested
        if _XFADE_TRANSITIONS.get(transition, transition) not in supported
    )
    if unsupported:
        names = ", ".join(unsupported)
        raise RenderError(
            f"Installed FFmpeg does not support xfade transition(s): {names}. "
            "Upgrade FFmpeg to a build with these xfade transitions."
        )


def _build_shot_render_command(
    ffmpeg: str,
    clips: list[tuple[Path, float, float, bool, str, float]],
    output: Path,
    width: int,
    height: int,
    frame_rate: float = 30.0,
    fit_mode: str = "crop",
    title_overlays: list[TitleOverlay] | None = None,
    asset_overlays: list[tuple[TitleOverlay, Path]] | None = None,
    crop_anchors: list[tuple[float, float]] | None = None,
    crop_paths: list[list[tuple[float, float, float]]] | None = None,
    zoom_paths: list[list[tuple[float, float, float]]] | None = None,
    music_path: Path | None = None,
    music_mode: str = "blend",
    music_volume: float = 0.22,
    music_fade_in: float = 1.0,
    music_fade_out: float = 1.5,
    music_ducking: bool = False,
    music_loudness_normalization: bool = False,
    music_target_lufs: float = -16.0,
    music_true_peak_db: float = -1.5,
    music_loudness_range: float = 11.0,
) -> list[str]:
    """Build a filter graph that preserves shot order and applies approved transitions."""
    if fit_mode not in {"crop", "pad"}:
        raise ValueError("fit_mode must be crop or pad")
    command = [ffmpeg, "-hide_banner", "-nostdin", "-y"]
    for source, *_ in clips:
        command.extend(["-i", str(source)])
    for overlay, asset_path in asset_overlays or []:
        if overlay.asset_format == "webm":
            command.extend(["-stream_loop", "-1", "-i", str(asset_path)])
        else:
            command.extend(
                [
                    "-framerate",
                    str(overlay.asset_fps),
                    "-stream_loop",
                    "-1",
                    "-i",
                    str(asset_path),
                ]
            )
    music_input_index = append_music_input(command, music_path)

    filters: list[str] = []
    for index, (_, start, end, has_audio, _, _) in enumerate(clips):
        duration = end - start
        if crop_anchors is not None and len(crop_anchors) != len(clips):
            raise ValueError("crop_anchors must contain one anchor per clip")
        anchor_x, anchor_y = crop_anchors[index] if crop_anchors is not None else (0.5, 0.5)
        if not (0 <= anchor_x <= 1 and 0 <= anchor_y <= 1):
            raise ValueError("crop anchors must be normalized between 0 and 1")
        if crop_paths is not None and len(crop_paths) != len(clips):
            raise ValueError("crop_paths must contain one path per clip")
        path = crop_paths[index] if crop_paths is not None else []
        if zoom_paths is not None and len(zoom_paths) != len(clips):
            raise ValueError("zoom_paths must contain one path per clip")
        zoom_path = zoom_paths[index] if zoom_paths is not None else []
        if path:
            if any(not (0 <= x <= 1 and 0 <= y <= 1) or timestamp < 0 for timestamp, x, y in path):
                raise ValueError(
                    "crop path timestamps and coordinates must be non-negative and normalized"
                )
            x_position = _crop_path_expression(path, 1)
            y_position = _crop_path_expression(path, 2)
        else:
            x_position = f"{anchor_x:.6f}"
            y_position = f"{anchor_y:.6f}"
        fit_filters = (
            f"scale=w={width}:h={height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}:"
            f"x='max(0,min(iw-ow,iw*({x_position})-ow/2))':"
            f"y='max(0,min(ih-oh,ih*({y_position})-oh/2))'"
            if fit_mode == "crop"
            else f"scale=w={width}:h={height}:force_original_aspect_ratio=decrease,"
            f"pad=width={width}:height={height}:x=(ow-iw)/2:y=(oh-ih)/2"
        )
        filters.append(
            f"[{index}:v:0]trim=start={_seconds(start)}:end={_seconds(end)},"
            f"setpts=PTS-STARTPTS,{fit_filters},"
            f"setsar=1,fps={_seconds(frame_rate)},settb=AVTB,"
            f"{_beat_zoom_filter(zoom_path, width, height)}"
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
            transition_name = _XFADE_TRANSITIONS[transition]
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

    current_video = _append_title_overlay_filters(
        filters, title_overlays, width, height, current_video
    )
    if asset_overlays:
        current_video = _append_asset_overlay_filters(
            filters, asset_overlays, width, height, current_video, len(clips)
        )
    filters.append(f"[{current_video}]format=yuv420p[vfinal]")
    filters.append(f"[{current_audio}]aresample=48000[baseaudio]")
    audio_output = append_music_mix(
        filters,
        base_label="baseaudio",
        music_input_index=music_input_index,
        duration_seconds=sum(end - start for _, start, end, *_ in clips)
        - sum(duration for *_, transition, duration in clips[1:] if transition != "cut"),
        mode=music_mode,
        volume=music_volume,
        fade_in_seconds=music_fade_in,
        fade_out_seconds=music_fade_out,
        ducking=music_ducking,
        loudness_normalization=music_loudness_normalization,
        target_lufs=music_target_lufs,
        true_peak_db=music_true_peak_db,
        loudness_range=music_loudness_range,
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[vfinal]",
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


def _probe_approved_clip_inputs(
    root: Path,
    approved: list,
    ffprobe: str,
) -> tuple[
    dict[Path, tuple[dict, float]],
    list[tuple[Path, float, float, bool, str, float]],
    set[Path],
]:
    """Validate and probe approved sources before any render output is created."""
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

    return source_cache, clip_inputs, source_paths


def _resolve_shot_output_dimensions(
    root: Path,
    clip_inputs: list[tuple[Path, float, float, bool, str, float]],
    source_cache: dict[Path, tuple[dict, float]],
    output_aspect_ratio: str | None,
    output_resolution: str | None,
) -> tuple[int, int]:
    """Resolve a valid even-sized output profile from source metadata or explicit options."""
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
    if output_aspect_ratio is None and output_resolution is None:
        width, height = _resolve_output_dimensions(root, width, height)
    else:
        width, height = _requested_output_dimensions(output_aspect_ratio, output_resolution)

    return width, height


def _render_and_validate_shot_output(
    *,
    root: Path,
    timeline_file: Path,
    timeline: ProjectDraftTimeline,
    approved: list,
    approved_titles: list[TitleOverlay],
    asset_overlays: list[tuple[TitleOverlay, Path]],
    selected_music: Path | None,
    music_mode: str,
    music_volume: float,
    music_fade_in: float,
    music_fade_out: float,
    music_ducking: bool,
    music_loudness_normalization: bool,
    music_target_lufs: float,
    music_true_peak_db: float,
    music_loudness_range: float,
    ffmpeg: str,
    ffprobe: str,
    destination: Path,
    width: int,
    height: int,
    expected_duration: float,
    timeout_seconds: float,
    dry_run: bool,
    fit_mode: str,
    reframe_mode: str,
    crop_anchors_path: Path | None,
    beat_punch_in: bool,
    clip_inputs: list[tuple[Path, float, float, bool, str, float]],
) -> tuple[dict, dict, dict, float] | None:
    """Build, execute, and verify FFmpeg output without registering project artifacts."""
    temporary_path = destination
    if not dry_run:
        temporary_file = tempfile.NamedTemporaryFile(
            prefix=".video-agent-shot-render-", suffix=".mp4", dir=destination.parent, delete=False
        )
        temporary_path = Path(temporary_file.name)
        temporary_file.close()
    try:
        try:
            crop_paths = detect_face_crop_paths(clip_inputs) if reframe_mode == "face" else None
            crop_anchors = None
            zoom_paths = (
                _build_beat_zoom_paths(root, selected_music, clip_inputs)
                if beat_punch_in and fit_mode == "crop"
                else None
            )
            if crop_anchors_path is not None:
                manual_anchors = load_crop_anchors(root, timeline_file, timeline, crop_anchors_path)
                if crop_paths is None:
                    crop_anchors = [(0.5, 0.5) for _ in approved]
                for index, clip in enumerate(approved):
                    if clip.clip_id in manual_anchors:
                        if crop_anchors is None:
                            crop_anchors = [(0.5, 0.5) for _ in approved]
                        crop_anchors[index] = manual_anchors[clip.clip_id]
                        if crop_paths is not None:
                            crop_paths[index] = []
        except (RuntimeError, ValueError, OSError) as exc:
            raise RenderError(str(exc)) from exc
        command = _build_shot_render_command(
            ffmpeg,
            clip_inputs,
            temporary_path,
            width,
            height,
            fit_mode=fit_mode,
            crop_anchors=crop_anchors,
            crop_paths=crop_paths,
            zoom_paths=zoom_paths,
            title_overlays=approved_titles,
            asset_overlays=asset_overlays,
            music_path=selected_music,
            music_mode=music_mode,
            music_volume=music_volume,
            music_fade_in=music_fade_in,
            music_fade_out=music_fade_out,
            music_ducking=music_ducking,
            music_loudness_normalization=music_loudness_normalization,
            music_target_lufs=music_target_lufs,
            music_true_peak_db=music_true_peak_db,
            music_loudness_range=music_loudness_range,
        )
        if dry_run:
            _validate_filter_graph(command, timeout=min(timeout_seconds, 120))
            return None
        run_ffmpeg(
            command,
            timeout=timeout_seconds,
            label="FFmpeg shot render",
            error_type=RenderError,
        )
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

    return output_probe, output_video, output_audio, actual_duration


def render_approved_shot_timeline(
    folder: Path,
    timeline_path: Path | None = None,
    output_path: Path | None = None,
    *,
    overwrite: bool = False,
    timeout_seconds: float = 3600,
    dry_run: bool = False,
    fit_mode: str = "crop",
    title_overlay_path: Path | None = None,
    music_path: Path | None = None,
    music_mode: str = "blend",
    music_volume: float = 0.22,
    music_fade_in: float = 1.0,
    music_fade_out: float = 1.5,
    music_ducking: bool = False,
    music_loudness_normalization: bool = False,
    music_target_lufs: float = -16.0,
    music_true_peak_db: float = -1.5,
    music_loudness_range: float = 11.0,
    output_aspect_ratio: str | None = None,
    output_resolution: str | None = None,
    reframe_mode: str = "center",
    crop_anchors_path: Path | None = None,
    beat_punch_in: bool = False,
) -> RenderReport | None:
    """Render only approved shots after every proposed shot has a decision."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    if fit_mode not in {"crop", "pad"}:
        raise ValueError("fit_mode must be crop or pad")
    if reframe_mode not in {"center", "face"}:
        raise ValueError("reframe_mode must be center or face")
    if (reframe_mode == "face" or crop_anchors_path is not None) and fit_mode != "crop":
        raise ValueError("face reframing and manual crop anchors require fit_mode='crop'")
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project folder does not exist: {root}")
    selected_music = (
        None if music_mode == "original" else resolve_selected_music(root, music_path=music_path)
    )

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
    timeline = apply_user_overrides(timeline)
    if timeline.rendering_performed:
        raise RenderError("optimized timeline must be a review artifact, not a render report")
    title_document = None
    title_file = None
    if title_overlay_path is not None:
        from .title_overlays import load_title_overlays

        title_document, title_file = load_title_overlays(root, title_overlay_path)
        if Path(title_document.source_timeline).resolve() != timeline_file:
            source_timeline = Path(title_document.source_timeline)
            if not source_timeline.is_absolute():
                source_timeline = root / source_timeline
            if source_timeline.resolve() != timeline_file:
                raise RenderError("title overlays reference a different optimized timeline")
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
    if destination in {
        timeline_file,
        root / ".video-agent" / "optimized-timeline.json",
        title_file,
    }:
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

    source_cache, clip_inputs, source_paths = _probe_approved_clip_inputs(root, approved, ffprobe)
    _validate_xfade_transitions(ffmpeg, clip_inputs)
    if destination in source_paths:
        raise RenderError("render output cannot overwrite a source video")
    if selected_music is not None and destination == selected_music:
        raise RenderError("render output cannot overwrite the selected soundtrack")
    width, height = _resolve_shot_output_dimensions(
        root, clip_inputs, source_cache, output_aspect_ratio, output_resolution
    )

    expected_duration = sum(end - start for _, start, end, *_ in clip_inputs) - sum(
        transition_duration
        for *_, transition, transition_duration in clip_inputs[1:]
        if transition != "cut"
    )
    if expected_duration <= 0:
        raise RenderError("approved shot sequence has no positive output duration")
    approved_titles = [
        item
        for item in (title_document.overlays if title_document is not None else [])
        if item.decision == "approved"
    ]
    for title in approved_titles:
        if title.end_seconds > expected_duration + 0.01:
            raise RenderError(
                f"title overlay {title.overlay_id} exceeds rendered timeline duration"
            )
    try:
        approved_assets = resolve_asset_overlays(root, approved_titles)
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise RenderError(str(exc)) from exc
    for asset_overlay, _asset_path in approved_assets:
        if asset_overlay.end_seconds > expected_duration + 0.01:
            raise RenderError(
                f"asset overlay {asset_overlay.overlay_id} exceeds rendered timeline duration"
            )
    dependencies = [timeline_file, *source_paths]
    if title_file is not None:
        dependencies.append(title_file)
    for asset_overlay, asset_path in approved_assets:
        if asset_overlay.asset_format == "webm":
            dependencies.append(asset_path)
        else:
            frame_pattern = re.sub(r"%0\d+d", "*", asset_path.name)
            dependencies.extend(asset_path.parent.glob(frame_pattern))
    if selected_music is not None:
        dependencies.append(selected_music)
    music_analysis_file = root / ".video-agent" / "music-analysis.json"
    if beat_punch_in and music_analysis_file.is_file():
        dependencies.append(music_analysis_file)
    plan_file = root / ".video-agent" / "video-project-plan.json"
    if plan_file.is_file():
        dependencies.append(plan_file)
    if crop_anchors_path is not None:
        dependencies.append(
            crop_anchors_path if crop_anchors_path.is_absolute() else root / crop_anchors_path
        )

    # Font configuration affects drawtext output even when no separate font file is tracked.
    from ..config import DEFAULT_TITLE_FONT, get_setting
    from .overlay_filters import _BUNDLED_FONT

    configured_font = (get_setting("TITLE_FONT", DEFAULT_TITLE_FONT) or DEFAULT_TITLE_FONT).strip()
    if approved_titles:
        if configured_font.casefold() in {"lato", "bundled"} and _BUNDLED_FONT.is_file():
            dependencies.append(_BUNDLED_FONT)
        else:
            configured_font_path = Path(configured_font).expanduser()
            if configured_font_path.is_file():
                dependencies.append(configured_font_path.resolve())

    ffmpeg_version_result = run_ffmpeg(
        [ffmpeg, "-version"],
        timeout=30,
        label="FFmpeg version for render cache",
        error_type=RenderError,
    )
    ffmpeg_version = (
        ffmpeg_version_result.stdout or ffmpeg_version_result.stderr or ""
    ).splitlines()
    render_config = {
        "cache_schema": 2,
        "title_font": configured_font,
        "ffmpeg_version": ffmpeg_version[0] if ffmpeg_version else "unknown",
        "width": width,
        "height": height,
        "expected_duration": expected_duration,
        "fit_mode": fit_mode,
        "reframe_mode": reframe_mode,
        "crop_anchors_path": str(crop_anchors_path) if crop_anchors_path else None,
        "beat_punch_in": beat_punch_in,
        "music_mode": music_mode,
        "music_volume": music_volume,
        "music_fade_in": music_fade_in,
        "music_fade_out": music_fade_out,
        "music_ducking": music_ducking,
        "music_loudness_normalization": music_loudness_normalization,
        "music_target_lufs": music_target_lufs,
        "music_true_peak_db": music_true_peak_db,
        "music_loudness_range": music_loudness_range,
        "ffmpeg": ffmpeg,
        "approved_clip_inputs": [
            (str(path), start, end, has_audio, transition, transition_duration)
            for path, start, end, has_audio, transition, transition_duration in clip_inputs
        ],
    }
    from .render_cache import (
        build_render_cache_key,
        restore_cached_render,
        store_render_cache,
    )

    cache_key = build_render_cache_key(
        dependencies,
        render_config,
        fingerprint_index_path=(root / ".video-agent" / "render-cache" / "file-fingerprints.json"),
    )
    cache_file = root / ".video-agent" / "render-cache" / f"{cache_key}.mp4"
    render_result = None
    if not dry_run and restore_cached_render(cache_file, destination):
        try:
            cached_probe = _run_probe(ffprobe, destination)
            cached_streams = cached_probe.get("streams", [])
            cached_video = next(
                (item for item in cached_streams if item.get("codec_type") == "video"), None
            )
            cached_audio = next(
                (item for item in cached_streams if item.get("codec_type") == "audio"), None
            )
            cached_duration = float(cached_probe.get("format", {}).get("duration") or 0)
            if (
                cached_video is not None
                and cached_audio is not None
                and cached_video.get("codec_name") == "h264"
                and cached_audio.get("codec_name") == "aac"
                and (int(cached_video.get("width") or 0), int(cached_video.get("height") or 0))
                == (width, height)
                and cached_duration > 0
                and abs(cached_duration - expected_duration)
                <= max(0.75, len(clip_inputs) / 30.0 * 2)
            ):
                render_result = (cached_probe, cached_video, cached_audio, cached_duration)
            else:
                destination.unlink(missing_ok=True)
        except (RenderError, OSError, ValueError, TypeError):
            destination.unlink(missing_ok=True)

    if render_result is None:
        render_result = _render_and_validate_shot_output(
            root=root,
            timeline_file=timeline_file,
            timeline=timeline,
            approved=approved,
            approved_titles=approved_titles,
            asset_overlays=approved_assets,
            selected_music=selected_music,
            music_mode=music_mode,
            music_volume=music_volume,
            music_fade_in=music_fade_in,
            music_fade_out=music_fade_out,
            music_ducking=music_ducking,
            music_loudness_normalization=music_loudness_normalization,
            music_target_lufs=music_target_lufs,
            music_true_peak_db=music_true_peak_db,
            music_loudness_range=music_loudness_range,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            destination=destination,
            width=width,
            height=height,
            expected_duration=expected_duration,
            timeout_seconds=timeout_seconds,
            dry_run=dry_run,
            fit_mode=fit_mode,
            reframe_mode=reframe_mode,
            crop_anchors_path=crop_anchors_path,
            beat_punch_in=beat_punch_in,
            clip_inputs=clip_inputs,
        )
        if render_result is not None:
            try:
                store_render_cache(destination, cache_file)
            except OSError:
                # A cache write failure must not discard an otherwise valid render.
                pass

    if render_result is None:
        return None
    output_probe, output_video, output_audio, actual_duration = render_result
    register_artifact(
        root,
        destination,
        dependencies=dependencies,
        config=render_config,
        verify_content=False,
    )

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
