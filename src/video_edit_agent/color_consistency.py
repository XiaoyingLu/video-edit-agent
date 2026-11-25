"""Objective, review-only color consistency analysis using FFmpeg signal statistics."""

from __future__ import annotations

import re
import shutil
import statistics
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .artifact_tracking import register_artifact, validate_artifact
from .media_runner import MediaError, run_ffmpeg
from .models import ProjectAnalysis


class ColorAnalysisError(MediaError):
    """Raised when source media cannot be sampled for color analysis."""


class ColorSample(BaseModel):
    """Average luma and chroma values measured from one representative frame."""

    model_config = ConfigDict(extra="forbid")

    timestamp_seconds: float = Field(ge=0)
    luma: float = Field(ge=0, le=255)
    chroma_u: float = Field(ge=0, le=255)
    chroma_v: float = Field(ge=0, le=255)


class ColorCorrectionSuggestion(BaseModel):
    """Bounded, non-destructive correction values relative to the project baseline."""

    model_config = ConfigDict(extra="forbid")

    luma_offset: float = Field(ge=-24, le=24)
    chroma_u_offset: float = Field(ge=-12, le=12)
    chroma_v_offset: float = Field(ge=-12, le=12)
    rationale: str


class VideoColorAnalysis(BaseModel):
    """Measured color consistency for one source video."""

    model_config = ConfigDict(extra="forbid")

    video_path: str
    baseline_luma: float = Field(ge=0, le=255)
    baseline_chroma_u: float = Field(ge=0, le=255)
    baseline_chroma_v: float = Field(ge=0, le=255)
    samples: list[ColorSample]
    suggestions: list[ColorCorrectionSuggestion]


class ProjectColorAnalysis(BaseModel):
    """Review-only color analysis report for a project."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    project_path: str
    source_analysis: str = ".video-agent/analysis.json"
    sample_count_per_video: int = Field(ge=1, le=12)
    videos: list[VideoColorAnalysis]
    approval_required: bool = True
    corrections_applied: bool = False
    rendering_performed: bool = False


def _sample_signalstats(ffmpeg: str, source: Path, timestamp: float) -> ColorSample:
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-vf",
        "signalstats,metadata=print:file=-",
        "-f",
        "null",
        "-",
    ]
    result = run_ffmpeg(
        command,
        timeout=60,
        label=f"Color sampling at {timestamp:.2f}s in {source}",
        error_type=ColorAnalysisError,
    )
    values: dict[str, float] = {}
    for key in ("YAVG", "UAVG", "VAVG"):
        match = re.search(rf"lavfi\.signalstats\.{key}=([0-9]+(?:\.[0-9]+)?)", result.stdout)
        if match is None:
            raise ColorAnalysisError(
                f"FFmpeg did not return {key} color statistics for {source} at {timestamp:.2f}s"
            )
        values[key] = float(match.group(1))
    return ColorSample(
        timestamp_seconds=timestamp,
        luma=values["YAVG"],
        chroma_u=values["UAVG"],
        chroma_v=values["VAVG"],
    )


def _bounded_offset(target: float, measured: float, limit: float) -> float:
    return round(max(-limit, min(limit, target - measured)), 2)


def analyze_project_color(
    folder: Path,
    *,
    sample_count: int = 5,
    analysis_path: Path | None = None,
) -> ProjectColorAnalysis:
    """Sample each source video and suggest conservative offsets from its own median baseline."""
    if not 1 <= sample_count <= 12:
        raise ValueError("sample_count must be between 1 and 12")
    root = folder.expanduser().resolve()
    source_analysis = (
        (analysis_path or root / ".video-agent" / "analysis.json").expanduser().resolve()
    )
    try:
        source_analysis.relative_to(root)
    except ValueError as exc:
        raise ValueError("analysis file must be inside the project folder") from exc
    if not source_analysis.is_file():
        raise FileNotFoundError(f"Run 'video-agent analyze {root}' before color analysis.")
    validate_artifact(root, source_analysis)
    analysis = ProjectAnalysis.model_validate_json(source_analysis.read_text(encoding="utf-8"))
    if Path(analysis.project_path).resolve() != root:
        raise ValueError("analysis artifact belongs to a different project folder")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ColorAnalysisError("ffmpeg was not found; install FFmpeg and ensure it is on PATH.")

    sampled_videos: list[tuple[str, list[ColorSample]]] = []
    all_samples: list[ColorSample] = []
    for video in analysis.videos:
        if video.duration_seconds <= 0 or not video.has_video:
            continue
        relative = Path(video.path)
        if relative.is_absolute():
            raise ValueError(f"Video path must be project-relative: {video.path}")
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Video path escapes the project folder: {video.path}") from exc
        if not source.is_file():
            raise FileNotFoundError(f"Source video does not exist: {source}")
        count = min(sample_count, max(1, int(video.duration_seconds)))
        timestamps = [video.duration_seconds * (index + 1) / (count + 1) for index in range(count)]
        samples = [_sample_signalstats(ffmpeg, source, timestamp) for timestamp in timestamps]
        sampled_videos.append((video.path, samples))
        all_samples.extend(samples)

    if not all_samples:
        raise ColorAnalysisError("No video frames were available for color analysis.")
    baseline_y = statistics.median(item.luma for item in all_samples)
    baseline_u = statistics.median(item.chroma_u for item in all_samples)
    baseline_v = statistics.median(item.chroma_v for item in all_samples)
    videos: list[VideoColorAnalysis] = []
    for video_path, samples in sampled_videos:
        suggestions = []
        for sample in samples:
            offsets = (
                _bounded_offset(baseline_y, sample.luma, 24),
                _bounded_offset(baseline_u, sample.chroma_u, 12),
                _bounded_offset(baseline_v, sample.chroma_v, 12),
            )
            if any(abs(value) >= 2 for value in offsets):
                rationale = (
                    "Suggested offsets move this sample toward the project-wide median "
                    "luma and chroma measured across all source videos."
                )
            else:
                rationale = "This sample is close to the project-wide measured baseline."
            suggestions.append(
                ColorCorrectionSuggestion(
                    luma_offset=offsets[0],
                    chroma_u_offset=offsets[1],
                    chroma_v_offset=offsets[2],
                    rationale=rationale,
                )
            )
        videos.append(
            VideoColorAnalysis(
                video_path=video_path,
                baseline_luma=baseline_y,
                baseline_chroma_u=baseline_u,
                baseline_chroma_v=baseline_v,
                samples=samples,
                suggestions=suggestions,
            )
        )
    return ProjectColorAnalysis(
        project_path=str(root),
        source_analysis=source_analysis.relative_to(root).as_posix(),
        sample_count_per_video=sample_count,
        videos=videos,
    )


def save_project_color_analysis(
    folder: Path,
    report: ProjectColorAnalysis,
    output_path: Path | None = None,
) -> Path:
    """Save the report without overwriting inputs or an existing review artifact."""
    from .atomic_io import write_json_atomic

    root = folder.expanduser().resolve()
    destination = (
        (output_path or root / ".video-agent" / "color-analysis.json").expanduser().resolve()
    )
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("color report output must be inside the project folder") from exc
    source = root / report.source_analysis
    if destination in {source}:
        raise ValueError("color report cannot overwrite its source analysis")
    if destination.exists():
        raise FileExistsError(f"Color report already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(destination, report)
    register_artifact(
        root,
        destination,
        dependencies=[source],
        config={"sample_count_per_video": report.sample_count_per_video},
    )
    return destination
