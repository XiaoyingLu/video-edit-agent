"""Evidence-based technical and structural critique of rendered video drafts."""

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import cv2
from pydantic import BaseModel, ConfigDict, Field

from ..artifact_tracking import validate_artifact
from ..config import get_setting
from ..editing.planning_schemas import VideoProjectPlan
from ..editing.timeline_schemas import ProjectDraftTimeline
from ..media_runner import MediaError, run_ffmpeg


class DraftCritiqueError(MediaError):
    """Raised when a rendered draft cannot be safely inspected."""


class FrameQualitySample(BaseModel):
    """Objective image measurements from one uniformly sampled output frame."""

    timestamp_seconds: float = Field(ge=0)
    mean_luma: float = Field(ge=0, le=255)
    dark_pixel_ratio: float = Field(ge=0, le=1)
    bright_pixel_ratio: float = Field(ge=0, le=1)
    sharpness_variance: float = Field(ge=0)


class AudioQualitySummary(BaseModel):
    """Audio stream presence, loudness measurements, and detected quiet intervals."""

    has_audio: bool
    mean_volume_db: float | None = None
    max_volume_db: float | None = None
    silence_intervals: list[tuple[float, float]] = Field(default_factory=list)


class CritiqueFinding(BaseModel):
    """One actionable observation with supporting measurements."""

    model_config = ConfigDict(extra="forbid")

    severity: Literal["info", "warning"]
    category: Literal["duration", "format", "pacing", "visual", "audio", "render", "story"]
    message: str
    evidence: str


class SemanticFrameReview(BaseModel):
    """Vision-model assessment of one rendered shot against its storyboard intent."""

    shot_id: str
    timestamp_seconds: float = Field(ge=0)
    match: Literal["match", "partial", "mismatch", "uncertain"]
    score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    evidence: str = Field(max_length=240)
    suggestion: str = Field(max_length=240)


class DraftCritiqueReport(BaseModel):
    """Read-only critique report for a rendered draft and its creative inputs."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    project_path: str
    rendered_video: str
    source_timeline: str
    source_plan: str
    duration_seconds: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_rate: float | None = Field(default=None, gt=0)
    video_codec: str
    audio_codec: str | None = None
    frames: list[FrameQualitySample] = Field(default_factory=list)
    audio: AudioQualitySummary
    findings: list[CritiqueFinding] = Field(default_factory=list)
    semantic_model: str | None = None
    semantic_reviews: list[SemanticFrameReview] = Field(default_factory=list)
    review_required: Literal[True] = True
    timeline_modified: Literal[False] = False
    rendered_video_modified: Literal[False] = False


def _run(command: list[str], *, timeout: float = 120) -> Any:
    return run_ffmpeg(
        command,
        timeout=timeout,
        label=f"Media inspection ({Path(command[0]).name})",
        error_type=DraftCritiqueError,
    )


def _safe_project_file(root: Path, path: Path, label: str) -> Path:
    candidate = path.expanduser()
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DraftCritiqueError(f"{label} must be inside the project folder.") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _sample_frames(
    ffmpeg: str, video: Path, duration: float, sample_count: int
) -> list[FrameQualitySample]:
    samples: list[FrameQualitySample] = []
    with tempfile.TemporaryDirectory(prefix="video-agent-critique-") as temporary:
        directory = Path(temporary)
        for index in range(sample_count):
            timestamp = duration * (index + 1) / (sample_count + 1)
            frame_path = directory / f"frame-{index + 1:02d}.jpg"
            result = _run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    str(video),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=640:-2:force_original_aspect_ratio=decrease",
                    "-q:v",
                    "4",
                    "-y",
                    str(frame_path),
                ],
                timeout=60,
            )
            if result.returncode != 0 or not frame_path.is_file():
                raise DraftCritiqueError(
                    f"Could not sample rendered frame at {timestamp:.2f}s: {result.stderr.strip()}"
                )
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise DraftCritiqueError(f"Could not decode sampled frame at {timestamp:.2f}s.")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            samples.append(
                FrameQualitySample(
                    timestamp_seconds=timestamp,
                    mean_luma=float(gray.mean()),
                    dark_pixel_ratio=float((gray < 16).mean()),
                    bright_pixel_ratio=float((gray > 239).mean()),
                    sharpness_variance=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                )
            )
    return samples


def _analyze_audio(ffmpeg: str, video: Path, has_audio: bool) -> AudioQualitySummary:
    if not has_audio:
        return AudioQualitySummary(has_audio=False)
    result = _run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(video),
            "-vn",
            "-af",
            "silencedetect=noise=-45dB:d=1.0,volumedetect",
            "-f",
            "null",
            "-",
        ],
        timeout=180,
    )
    if result.returncode != 0:
        detail = result.stderr.strip()[-1200:]
        raise DraftCritiqueError(f"Could not analyze rendered audio: {detail}")
    mean_match = re.search(r"mean_volume:\s*(-?inf|-?\d+(?:\.\d+)?)\s*dB", result.stderr)
    max_match = re.search(r"max_volume:\s*(-?inf|-?\d+(?:\.\d+)?)\s*dB", result.stderr)
    silence_starts = [
        float(value) for value in re.findall(r"silence_start:\s*(-?\d+(?:\.\d+)?)", result.stderr)
    ]
    silence_ends = [
        float(value) for value in re.findall(r"silence_end:\s*(-?\d+(?:\.\d+)?)", result.stderr)
    ]
    intervals = [
        (start, end)
        for start, end in zip(silence_starts, silence_ends, strict=False)
        if end > start
    ]
    return AudioQualitySummary(
        has_audio=True,
        mean_volume_db=(
            float(mean_match.group(1)) if mean_match and mean_match.group(1) != "-inf" else None
        ),
        max_volume_db=(
            float(max_match.group(1)) if max_match and max_match.group(1) != "-inf" else None
        ),
        silence_intervals=intervals,
    )


def _semantic_targets(
    clips: list,
    plan: VideoProjectPlan,
    *,
    rendered_duration: float,
    sample_limit: int,
) -> list[tuple[float, object, object | None]]:
    """Choose evenly distributed clip midpoints and match clips to storyboard shots."""
    shots = {shot.shot_id: shot for scene in plan.scenes for shot in scene.shots}
    targets: list[tuple[float, object, object | None]] = []
    cursor = 0.0
    previous_index: int | None = None
    for clip in clips:
        overlap = (
            clip.transition_duration_seconds
            if previous_index is not None
            and clip.sequence_index == previous_index + 1
            and clip.transition_in != "cut"
            else 0.0
        )
        start = max(0.0, cursor - overlap)
        timestamp = start + clip.duration_seconds / 2
        timestamp = min(max(timestamp, 0.0), max(0.0, rendered_duration - 0.05))
        targets.append((timestamp, clip, shots.get(clip.clip_id)))
        cursor = start + clip.duration_seconds
        previous_index = clip.sequence_index

    if len(targets) <= sample_limit:
        return targets
    if sample_limit == 1:
        return [targets[len(targets) // 2]]
    indices = sorted(
        {round(index * (len(targets) - 1) / (sample_limit - 1)) for index in range(sample_limit)}
    )
    return [targets[index] for index in indices]


def _review_semantic_frames(
    ffmpeg: str,
    video: Path,
    duration: float,
    clips: list,
    plan: VideoProjectPlan,
    *,
    sample_limit: int,
    model: str | None,
) -> tuple[str, list[SemanticFrameReview]]:
    """Compare representative rendered frames with storyboard intent using local Ollama."""
    from ..understanding.visual_analyzer import OllamaVisionAnalyzer, VisionModelError

    selected_model = model or get_setting("OLLAMA_MODEL")
    analyzer = OllamaVisionAnalyzer(model=selected_model)
    reviews: list[SemanticFrameReview] = []
    targets = _semantic_targets(
        clips,
        plan,
        rendered_duration=duration,
        sample_limit=sample_limit,
    )
    try:
        with tempfile.TemporaryDirectory(prefix="video-agent-semantic-critique-") as temporary:
            directory = Path(temporary)
            for index, (timestamp, clip, shot) in enumerate(targets, start=1):
                frame_path = directory / f"semantic-frame-{index:02d}.jpg"
                result = _run(
                    [
                        ffmpeg,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-nostdin",
                        "-ss",
                        f"{timestamp:.3f}",
                        "-i",
                        str(video),
                        "-frames:v",
                        "1",
                        "-vf",
                        "scale=768:-2:force_original_aspect_ratio=decrease",
                        "-q:v",
                        "3",
                        "-y",
                        str(frame_path),
                    ],
                    timeout=60,
                )
                if result.returncode != 0 or not frame_path.is_file():
                    raise DraftCritiqueError(
                        f"Could not extract semantic review frame at {timestamp:.2f}s: "
                        f"{result.stderr.strip()}"
                    )
                if shot is None:
                    purpose = (
                        clip.sequence_reason or "Match the rendered shot to its timeline intent"
                    )
                    visual_description = (
                        clip.scene_summary or "No storyboard description is available"
                    )
                    shot_id = clip.clip_id
                    primary_subject = clip.primary_subject
                    expected_role = clip.visual_role
                    expected_scale = clip.shot_scale
                else:
                    purpose = shot.purpose
                    visual_description = shot.visual_description
                    shot_id = shot.shot_id
                    primary_subject = shot.primary_subject
                    expected_role = shot.visual_role
                    expected_scale = shot.shot_scale
                try:
                    assessment = analyzer.compare_frame_to_storyboard(
                        frame_path,
                        purpose=purpose,
                        visual_description=visual_description,
                        primary_subject=primary_subject,
                        expected_role=expected_role,
                        expected_scale=expected_scale,
                    )
                    reviews.append(
                        SemanticFrameReview(
                            shot_id=shot_id,
                            timestamp_seconds=timestamp,
                            **assessment,
                        )
                    )
                except VisionModelError as exc:
                    raise DraftCritiqueError(
                        f"Semantic critique failed for shot {shot_id}: {exc}"
                    ) from exc
    finally:
        analyzer.unload_model()
    return selected_model, reviews


def _semantic_critique_findings(
    semantic_reviews: list[SemanticFrameReview],
) -> list[CritiqueFinding]:
    """Turn storyboard comparison results into actionable findings."""
    findings: list[CritiqueFinding] = []
    concerns = [
        review
        for review in semantic_reviews
        if review.match in {"partial", "mismatch"} and review.confidence >= 0.55
    ]
    for review in concerns:
        severity = "warning" if review.match == "mismatch" else "info"
        findings.append(
            CritiqueFinding(
                severity=severity,
                category="story",
                message=(
                    f"Rendered shot {review.shot_id} may not match its storyboard intent "
                    f"({review.match}, score {review.score:.2f})."
                ),
                evidence=(
                    f"At {review.timestamp_seconds:.1f}s, confidence "
                    f"{review.confidence:.2f}: {review.evidence} "
                    f"Suggested review: {review.suggestion}"
                ),
            )
        )
    return findings


def _quality_critique_findings(
    plan: VideoProjectPlan,
    timeline: ProjectDraftTimeline,
    duration: float,
    width: int,
    height: int,
    frames: list[FrameQualitySample],
    audio: AudioQualitySummary,
    semantic: bool,
    semantic_reviews: list[SemanticFrameReview],
) -> list[CritiqueFinding]:
    """Evaluate measurable render quality and timeline consistency without media I/O."""
    findings: list[CritiqueFinding] = []
    target_duration = plan.brief.target_duration_seconds
    duration_delta = duration - target_duration
    duration_tolerance = max(2.0, target_duration * 0.1)
    if abs(duration_delta) > duration_tolerance:
        findings.append(
            CritiqueFinding(
                severity="warning",
                category="duration",
                message="Rendered duration differs noticeably from the creative brief.",
                evidence=(
                    f"Brief: {target_duration:.2f}s; rendered: {duration:.2f}s "
                    f"({duration_delta:+.2f}s)."
                ),
            )
        )

    target_ratio = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0}[plan.brief.aspect_ratio]
    actual_ratio = width / height
    if abs(actual_ratio - target_ratio) / target_ratio > 0.03:
        findings.append(
            CritiqueFinding(
                severity="warning",
                category="format",
                message="Rendered aspect ratio does not match the creative brief.",
                evidence=(
                    f"Brief: {plan.brief.aspect_ratio}; rendered: {width}x{height} "
                    f"({actual_ratio:.3f}:1)."
                ),
            )
        )

    long_side = {"720p": 1280, "1080p": 1920, "4k": 3840}[plan.brief.resolution]
    short_side = {"720p": 720, "1080p": 1080, "4k": 2160}[plan.brief.resolution]
    if plan.brief.aspect_ratio == "16:9":
        minimum_width, minimum_height = long_side, short_side
    elif plan.brief.aspect_ratio == "9:16":
        minimum_width, minimum_height = short_side, long_side
    else:
        minimum_width = minimum_height = short_side
    if width < minimum_width or height < minimum_height:
        findings.append(
            CritiqueFinding(
                severity="warning",
                category="format",
                message="Rendered resolution is below the creative brief target.",
                evidence=(
                    f"Requested at least {minimum_width}x{minimum_height}; "
                    f"rendered {width}x{height}."
                ),
            )
        )

    approved_clips = [clip for clip in timeline.clips if clip.decision == "approved"]
    expected_timeline_duration = sum(clip.duration_seconds for clip in approved_clips) - sum(
        clip.transition_duration_seconds
        for previous, clip in zip(approved_clips, approved_clips[1:], strict=False)
        if clip.sequence_index == previous.sequence_index + 1 and clip.transition_in != "cut"
    )
    if abs(duration - expected_timeline_duration) > max(0.5, duration * 0.02):
        findings.append(
            CritiqueFinding(
                severity="warning",
                category="render",
                message="Rendered duration does not closely match the approved timeline.",
                evidence=(
                    f"Timeline estimate: {expected_timeline_duration:.2f}s; "
                    f"rendered: {duration:.2f}s."
                ),
            )
        )

    pacing_outliers = [clip for clip in approved_clips if clip.pacing_fit in {"short", "long"}]
    if pacing_outliers:
        findings.append(
            CritiqueFinding(
                severity="info",
                category="pacing",
                message="Some approved shots were already outside their recommended pacing range.",
                evidence=(
                    f"{len(pacing_outliers)} of {len(approved_clips)} approved shots were marked "
                    f"short or long during timeline review: "
                    + ", ".join(
                        f"{clip.clip_id} ({clip.duration_seconds:.1f}s, {clip.pacing_fit})"
                        for clip in pacing_outliers[:6]
                    )
                    + "."
                ),
            )
        )

    if not audio.has_audio:
        findings.append(
            CritiqueFinding(
                severity="warning",
                category="audio",
                message="Rendered video has no audio stream.",
                evidence="ffprobe did not report an audio stream.",
            )
        )
    elif audio.mean_volume_db is None:
        findings.append(
            CritiqueFinding(
                severity="warning",
                category="audio",
                message=(
                    "Rendered audio appears silent or could not produce a measurable mean level."
                ),
                evidence="FFmpeg volumedetect did not report a finite mean volume.",
            )
        )
    long_silences = [(start, end) for start, end in audio.silence_intervals if end - start >= 3]
    if long_silences:
        findings.append(
            CritiqueFinding(
                severity="info",
                category="audio",
                message="Long quiet intervals were detected in the rendered soundtrack.",
                evidence=", ".join(f"{start:.1f}–{end:.1f}s" for start, end in long_silences[:6]),
            )
        )

    dark_frames = [frame for frame in frames if frame.dark_pixel_ratio >= 0.5]
    clipped_frames = [frame for frame in frames if frame.bright_pixel_ratio >= 0.5]
    if dark_frames:
        findings.append(
            CritiqueFinding(
                severity="info",
                category="visual",
                message="Some sampled frames contain extensive near-black areas.",
                evidence=", ".join(
                    f"{frame.timestamp_seconds:.1f}s ({frame.dark_pixel_ratio:.0%} below luma 16)"
                    for frame in dark_frames
                ),
            )
        )
    if clipped_frames:
        findings.append(
            CritiqueFinding(
                severity="info",
                category="visual",
                message="Some sampled frames contain extensive near-white areas.",
                evidence=", ".join(
                    f"{frame.timestamp_seconds:.1f}s "
                    f"({frame.bright_pixel_ratio:.0%} above luma 239)"
                    for frame in clipped_frames
                ),
            )
        )
    if not findings:
        findings.append(
            CritiqueFinding(
                severity="info",
                category="render",
                message=(
                    "No threshold-based or semantic issues were detected."
                    if semantic
                    else "No threshold-based issues were detected in this pass."
                ),
                evidence=(
                    f"Checked duration, aspect ratio, approved timeline consistency, "
                    f"{len(frames)} sampled frames, and audio stream levels."
                    + (
                        f" Semantic review compared {len(semantic_reviews)} storyboard shots."
                        if semantic
                        else ""
                    )
                ),
            )
        )

    return findings


def critique_rendered_draft(
    folder: Path,
    *,
    video_path: Path | None = None,
    timeline_path: Path | None = None,
    plan_path: Path | None = None,
    sample_count: int = 8,
    semantic: bool = False,
    semantic_samples: int = 8,
    vision_model: str | None = None,
) -> DraftCritiqueReport:
    """Inspect a rendered draft against its approved timeline and creative brief."""
    if not 1 <= sample_count <= 24:
        raise ValueError("sample_count must be between 1 and 24")
    if not 1 <= semantic_samples <= 12:
        raise ValueError("semantic_samples must be between 1 and 12")
    if vision_model is not None and not vision_model.strip():
        raise ValueError("vision_model cannot be empty")
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project folder does not exist: {root}")
    video = _safe_project_file(
        root,
        video_path or root / ".video-agent" / "rendered-draft.mp4",
        "Rendered video",
    )
    timeline_file = _safe_project_file(
        root,
        timeline_path or root / ".video-agent" / "optimized-timeline.json",
        "Approved timeline",
    )
    plan_file = _safe_project_file(
        root,
        plan_path or root / ".video-agent" / "video-project-plan.json",
        "Creative project plan",
    )
    for artifact in (video, timeline_file, plan_file):
        validate_artifact(root, artifact)

    timeline = ProjectDraftTimeline.model_validate_json(timeline_file.read_text(encoding="utf-8"))
    plan = VideoProjectPlan.model_validate_json(plan_file.read_text(encoding="utf-8"))
    if Path(timeline.project_path).resolve() != root or Path(plan.project_path).resolve() != root:
        raise DraftCritiqueError("Timeline and creative plan must belong to this project folder.")
    if not timeline.timeline_approved or any(clip.decision == "pending" for clip in timeline.clips):
        raise DraftCritiqueError("The source timeline must be fully reviewed and approved.")
    if not plan.scenes:
        raise DraftCritiqueError("The creative plan must contain a completed storyboard.")

    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if ffprobe is None or ffmpeg is None:
        raise DraftCritiqueError("ffmpeg and ffprobe must be installed and available on PATH.")
    probe = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate",
            "-of",
            "json",
            str(video),
        ],
        timeout=60,
    )
    if probe.returncode != 0:
        detail = probe.stderr.strip()
        raise DraftCritiqueError(f"ffprobe could not inspect the rendered video: {detail}")
    try:
        payload = json.loads(probe.stdout)
        duration = float(payload["format"]["duration"])
        streams = payload["streams"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DraftCritiqueError("ffprobe returned incomplete rendered-video metadata.") from exc
    video_stream = next((item for item in streams if item.get("codec_type") == "video"), None)
    if video_stream is None:
        raise DraftCritiqueError("Rendered output does not contain a video stream.")
    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    if video.suffix.lower() != ".mp4":
        raise DraftCritiqueError("Rendered video must use the .mp4 extension.")
    if duration <= 0 or width <= 0 or height <= 0:
        raise DraftCritiqueError("Rendered output has invalid duration or dimensions.")
    audio_stream = next((item for item in streams if item.get("codec_type") == "audio"), None)

    frames = _sample_frames(ffmpeg, video, duration, sample_count)
    audio = _analyze_audio(ffmpeg, video, audio_stream is not None)
    findings: list[CritiqueFinding] = []
    semantic_model = None
    semantic_reviews: list[SemanticFrameReview] = []
    if semantic:
        approved_clips_for_review = [clip for clip in timeline.clips if clip.decision == "approved"]
        semantic_model, semantic_reviews = _review_semantic_frames(
            ffmpeg,
            video,
            duration,
            approved_clips_for_review,
            plan,
            sample_limit=semantic_samples,
            model=vision_model,
        )
        findings.extend(_semantic_critique_findings(semantic_reviews))

    findings.extend(
        _quality_critique_findings(
            plan, timeline, duration, width, height, frames, audio, semantic, semantic_reviews
        )
    )

    return DraftCritiqueReport(
        project_path=str(root),
        rendered_video=video.relative_to(root).as_posix(),
        source_timeline=timeline_file.relative_to(root).as_posix(),
        source_plan=plan_file.relative_to(root).as_posix(),
        duration_seconds=duration,
        width=width,
        height=height,
        frame_rate=_parse_frame_rate(video_stream.get("avg_frame_rate")),
        video_codec=str(video_stream.get("codec_name") or "unknown"),
        audio_codec=str(audio_stream.get("codec_name")) if audio_stream else None,
        frames=frames,
        audio=audio,
        findings=findings,
        semantic_model=semantic_model,
        semantic_reviews=semantic_reviews,
    )


def _parse_frame_rate(value: str | None) -> float | None:
    """Parse ffprobe's rational frame-rate representation."""
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        numerator, denominator = value.split("/", maxsplit=1)
        rate = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate > 0 else None
