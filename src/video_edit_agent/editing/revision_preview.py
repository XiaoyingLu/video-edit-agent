"""Render explicitly approved timeline revisions as separate preview artifacts."""

from ..atomic_io import write_json_atomic
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..artifact_tracking import register_artifact, validate_artifact
from ..rendering.renderer import RenderReport
from ..rendering.shot_renderer import render_approved_shot_timeline
from .planning_schemas import VideoProjectPlan
from .timeline_schemas import ProjectDraftTimeline


class RevisionPreviewError(ValueError):
    """Raised when a revision preview cannot be created safely."""


class RevisionPreviewReport(BaseModel):
    """Provenance and render metadata for a separate revision preview."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    source_timeline_path: str
    preview_video_path: str
    render_report_path: str
    preview_resolution: str = "source"
    duration_seconds: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    video_codec: str
    audio_codec: str
    segment_count: int = Field(ge=1)
    source_video_count: int = Field(ge=0)
    timeline_approved: bool = True
    original_timeline_modified: bool = False
    source_media_modified: bool = False


def render_timeline_revision_preview(
    folder: Path,
    *,
    timeline_path: Path,
    output_path: Path | None = None,
    timeout_seconds: float = 3600,
    dry_run: bool = False,
    fit_mode: str = "crop",
    title_overlay_path: Path | None = None,
) -> tuple[RenderReport | None, Path | None]:
    """Render an approved revision as a fast 720p preview when a project profile exists."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project folder does not exist: {root}")

    timeline_file = timeline_path.expanduser()
    timeline_file = (
        (root / timeline_file).resolve()
        if not timeline_file.is_absolute()
        else timeline_file.resolve()
    )
    revision_directory = (root / ".video-agent" / "timeline-revisions").resolve()
    try:
        timeline_file.relative_to(revision_directory)
    except ValueError as exc:
        raise RevisionPreviewError(
            "preview rendering requires a timeline under .video-agent/timeline-revisions/"
        ) from exc
    if not timeline_file.is_file():
        raise FileNotFoundError(f"Missing revised timeline: {timeline_file}")
    if not validate_artifact(root, timeline_file):
        raise RevisionPreviewError("timeline revision is stale; regenerate it before rendering")

    timeline = ProjectDraftTimeline.model_validate_json(timeline_file.read_text(encoding="utf-8"))
    if Path(timeline.project_path).resolve() != root:
        raise RevisionPreviewError("timeline revision belongs to a different project folder")
    if not timeline.timeline_approved or any(clip.decision == "pending" for clip in timeline.clips):
        raise RevisionPreviewError(
            "timeline revision is not fully approved; run 'video-agent review-timeline' first"
        )

    preview_aspect_ratio = None
    preview_resolution = "source"
    plan_file = root / ".video-agent" / "video-project-plan.json"
    if plan_file.is_file():
        if not validate_artifact(root, plan_file):
            raise RevisionPreviewError(
                "video project plan is stale; regenerate it before previewing"
            )
        plan = VideoProjectPlan.model_validate_json(plan_file.read_text(encoding="utf-8"))
        if Path(plan.project_path).resolve() != root:
            raise RevisionPreviewError("video project plan belongs to a different project folder")
        preview_aspect_ratio = plan.brief.aspect_ratio
        preview_resolution = "720p"

    default_output = root / ".video-agent" / "revision-previews" / f"{timeline_file.stem}.mp4"
    destination = (output_path or default_output).expanduser()
    destination = (
        (root / destination).resolve() if not destination.is_absolute() else destination.resolve()
    )
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise RevisionPreviewError("preview output must remain inside the project folder") from exc
    if destination.suffix.lower() != ".mp4":
        raise RevisionPreviewError("preview output must use the .mp4 extension")
    manifest_file = destination.with_suffix(".preview.json")
    if not dry_run and (destination.exists() or manifest_file.exists()):
        raise FileExistsError(
            f"Revision preview already exists: {destination}; choose a new --output path"
        )

    render_options = {}
    if preview_aspect_ratio is not None:
        render_options = {
            "output_aspect_ratio": preview_aspect_ratio,
            "output_resolution": "720p",
        }
    report = render_approved_shot_timeline(
        root,
        timeline_path=timeline_file,
        output_path=destination,
        overwrite=False,
        timeout_seconds=timeout_seconds,
        dry_run=dry_run,
        fit_mode=fit_mode,
        title_overlay_path=title_overlay_path,
        **render_options,
    )
    if dry_run:
        return None, None
    if report is None:
        raise RevisionPreviewError("renderer returned no report for a completed preview")

    result = RevisionPreviewReport(
        source_timeline_path=timeline_file.relative_to(root).as_posix(),
        preview_video_path=destination.relative_to(root).as_posix(),
        render_report_path=manifest_file.relative_to(root).as_posix(),
        preview_resolution=preview_resolution,
        duration_seconds=report.actual_duration_seconds,
        width=report.width,
        height=report.height,
        video_codec=report.video_codec,
        audio_codec=report.audio_codec,
        segment_count=report.segment_count,
        source_video_count=report.source_video_count,
    )
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(manifest_file, result)
    dependencies = [timeline_file, destination]
    if title_overlay_path is not None:
        overlay = title_overlay_path.expanduser()
        overlay = (root / overlay).resolve() if not overlay.is_absolute() else overlay.resolve()
        dependencies.append(overlay)
    register_artifact(root, manifest_file, dependencies=dependencies)
    return report, manifest_file
