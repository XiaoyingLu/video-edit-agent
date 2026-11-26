"""Apply browser-authored sequence and trim edits as a separate timeline revision."""

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..artifact_tracking import register_artifact, validate_artifact
from ..atomic_io import write_json_atomic
from .timeline_schemas import ProjectDraftTimeline


class TimelineVisualEditError(ValueError):
    """Raised when browser-authored timeline edits are unsafe or stale."""


class TimelineVisualEditClip(BaseModel):
    """One clip's identity and proposed source interval in the desired sequence."""

    model_config = ConfigDict(extra="forbid")

    clip_id: str = Field(min_length=1)
    source_start_seconds: float = Field(ge=0)
    source_end_seconds: float = Field(gt=0)
    transition_in: (
        Literal["cut", "crossfade", "dissolve", "fadeblack", "wipeleft", "slideleft", "circleopen"]
        | None
    ) = None
    transition_duration_seconds: float | None = Field(default=None, ge=0, le=2)


class TimelineVisualEditExport(BaseModel):
    """Version-bound browser export describing a complete proposed sequence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    timeline_path: str
    timeline_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    clips: list[TimelineVisualEditClip]


def load_timeline_visual_edits(path: Path) -> TimelineVisualEditExport:
    """Load and validate a browser-exported sequence edit file."""
    try:
        result = TimelineVisualEditExport.model_validate_json(
            path.expanduser().read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise TimelineVisualEditError(f"Invalid visual timeline edits: {exc}") from exc
    if not result.clips:
        raise TimelineVisualEditError("Visual timeline edits must contain at least one clip")
    ids = [clip.clip_id for clip in result.clips]
    if len(ids) != len(set(ids)):
        raise TimelineVisualEditError("Visual timeline edits contain duplicate clip IDs")
    return result


def apply_timeline_visual_edits(
    folder: Path,
    edits: TimelineVisualEditExport,
    *,
    timeline_path: Path | None = None,
    minimum_clip_duration_seconds: float = 1.0,
) -> Path:
    """Create a new unapproved timeline with validated ordering and trims."""
    root = folder.expanduser().resolve()
    source_file = (timeline_path or root / edits.timeline_path).expanduser()
    source_file = (
        (root / source_file).resolve() if not source_file.is_absolute() else source_file.resolve()
    )
    try:
        source_file.relative_to(root)
    except ValueError as exc:
        raise TimelineVisualEditError(
            "Source timeline must stay inside the project folder"
        ) from exc
    if not source_file.is_file():
        raise FileNotFoundError(f"Missing source timeline: {source_file}")
    if not validate_artifact(root, source_file):
        raise TimelineVisualEditError("Source timeline is stale or untracked; regenerate it first")
    relative_path = source_file.relative_to(root).as_posix()
    if relative_path != edits.timeline_path:
        raise TimelineVisualEditError("Edit export targets a different timeline")
    current_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
    if current_hash != edits.timeline_sha256:
        raise TimelineVisualEditError(
            "Timeline changed after visual editing; regenerate the editor"
        )
    source = ProjectDraftTimeline.model_validate_json(source_file.read_text(encoding="utf-8"))
    if Path(source.project_path).resolve() != root:
        raise TimelineVisualEditError("Timeline belongs to a different project folder")
    original = {clip.clip_id: clip for clip in source.clips}
    requested_ids = [clip.clip_id for clip in edits.clips]
    if len(requested_ids) != len(original) or set(requested_ids) != set(original):
        raise TimelineVisualEditError(
            "Edit export must contain every timeline clip exactly once; "
            "clips cannot be added or removed"
        )
    if minimum_clip_duration_seconds <= 0:
        raise TimelineVisualEditError("Minimum clip duration must be positive")

    revised_clips = []
    for edit in edits.clips:
        if (edit.transition_in is None) != (edit.transition_duration_seconds is None):
            raise TimelineVisualEditError(
                "Transition type and duration must be supplied together"
            )
        if edit.transition_in == "cut" and edit.transition_duration_seconds != 0:
            raise TimelineVisualEditError("Cut transitions must have a zero duration")
        if edit.transition_in not in {None, "cut"} and (edit.transition_duration_seconds or 0) <= 0:
            raise TimelineVisualEditError("Non-cut transitions must have a positive duration")

    for index, edit in enumerate(edits.clips, start=1):
        original_clip = original[edit.clip_id]
        if (
            edit.source_start_seconds < original_clip.source_start_seconds - 0.001
            or edit.source_end_seconds > original_clip.source_end_seconds + 0.001
            or edit.source_end_seconds <= edit.source_start_seconds
        ):
            raise TimelineVisualEditError(
                f"Trim for {edit.clip_id} must stay within its original source interval "
                f"({original_clip.source_start_seconds:.3f}–"
                f"{original_clip.source_end_seconds:.3f}s)"
            )
        duration = round(edit.source_end_seconds - edit.source_start_seconds, 3)
        if duration < minimum_clip_duration_seconds:
            raise TimelineVisualEditError(
                f"Trim for {edit.clip_id} is shorter than the "
                f"{minimum_clip_duration_seconds:.2f}s minimum"
            )
        clip = original_clip.model_copy(deep=True)
        clip.sequence_index = index
        clip.source_start_seconds = round(edit.source_start_seconds, 3)
        clip.source_end_seconds = round(edit.source_end_seconds, 3)
        clip.duration_seconds = duration
        clip.decision = "pending"
        clip.review_required = True
        ratio = duration / clip.pacing_target_seconds
        clip.pacing_fit = "short" if ratio < 0.75 else "long" if ratio > 1.25 else "within"
        clip.pacing_note = (
            f"Visually edited shot is {duration:.1f}s against a "
            f"{clip.pacing_target_seconds:.1f}s pacing target; review the new timing."
        )
        revised_clips.append(clip)

    # Reordering changes adjacency, so old transition choices are no longer trustworthy.
    order_changed = requested_ids != [clip.clip_id for clip in source.clips]
    for index, clip in enumerate(revised_clips):
        edit = edits.clips[index]
        if index == 0:
            clip.transition_in = "cut"
            clip.transition_duration_seconds = 0
        elif edit.transition_in is not None:
            clip.transition_in = edit.transition_in
            overlap_limit = min(
                clip.duration_seconds * 0.25,
                revised_clips[index - 1].duration_seconds * 0.25,
            )
            clip.transition_duration_seconds = round(
                min(edit.transition_duration_seconds or 0, overlap_limit), 3
            )
            if clip.transition_duration_seconds <= 0:
                clip.transition_in = "cut"
                clip.transition_duration_seconds = 0
        elif order_changed:
            clip.transition_in = "cut"
            clip.transition_duration_seconds = 0
        elif clip.transition_in != "cut":
            overlap_limit = min(
                clip.duration_seconds * 0.25,
                revised_clips[index - 1].duration_seconds * 0.25,
            )
            clip.transition_duration_seconds = round(
                min(clip.transition_duration_seconds, overlap_limit), 3
            )
            if clip.transition_duration_seconds <= 0:
                clip.transition_in = "cut"
                clip.transition_duration_seconds = 0
    proposed_duration = round(
        sum(clip.duration_seconds for clip in revised_clips)
        - sum(clip.transition_duration_seconds for clip in revised_clips[1:]),
        3,
    )
    if proposed_duration > source.target_duration_seconds + 0.02:
        raise TimelineVisualEditError(
            "Edited sequence exceeds the requested target duration; shorten one or more clips"
        )
    revised = source.model_copy(
        update={
            "clips": revised_clips,
            "proposed_duration_seconds": proposed_duration,
            "approved_duration_seconds": 0,
            "timeline_approved": False,
        }
    )
    # Validate all cross-clip invariants after reindexing and transition reset.
    revised = ProjectDraftTimeline.model_validate(revised.model_dump())

    directory = root / ".video-agent" / "timeline-revisions"
    directory.mkdir(parents=True, exist_ok=True)
    number = 1
    while (directory / f"visual-edit-{number:03}.json").exists():
        number += 1
    destination = directory / f"visual-edit-{number:03}.json"
    write_json_atomic(destination, revised)
    register_artifact(root, destination, dependencies=[source_file])
    return destination
