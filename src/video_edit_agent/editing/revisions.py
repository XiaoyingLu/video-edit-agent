"""Safe, reviewable per-shot revisions for an approved draft timeline."""

from ..atomic_io import write_json_atomic
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..artifact_tracking import register_artifact, validate_artifact
from .timeline_schemas import ProjectDraftTimeline


class TimelineIntervalEdit(BaseModel):
    """A user-authored source interval for one existing timeline shot."""

    model_config = ConfigDict(extra="forbid")

    clip_id: str = Field(min_length=1)
    source_start_seconds: float = Field(ge=0)
    source_end_seconds: float = Field(gt=0)


class TimelineRevisionResult(BaseModel):
    """Provenance for a separate timeline revision."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    source_timeline_path: str
    revision_path: str
    changed_clip_ids: list[str] = Field(default_factory=list)
    before_duration_seconds: float = Field(ge=0)
    after_duration_seconds: float = Field(ge=0)
    review_required: bool = True
    timeline: ProjectDraftTimeline


class TimelineClipChange(BaseModel):
    """Before/after values for one changed timeline clip."""

    model_config = ConfigDict(extra="forbid")

    clip_id: str
    before_start_seconds: float
    after_start_seconds: float
    before_end_seconds: float
    after_end_seconds: float
    before_duration_seconds: float
    after_duration_seconds: float
    before_transition: str
    after_transition: str


class TimelineComparison(BaseModel):
    """Deterministic comparison of two timeline versions."""

    model_config = ConfigDict(extra="forbid")

    before_path: str
    after_path: str
    before_duration_seconds: float
    after_duration_seconds: float
    duration_delta_seconds: float
    changes: list[TimelineClipChange] = Field(default_factory=list)


class TimelineRevisionError(ValueError):
    """Raised when a proposed timeline revision is unsafe or invalid."""


def _timeline_duration(timeline: ProjectDraftTimeline) -> float:
    return round(
        sum(clip.duration_seconds for clip in timeline.clips)
        - sum(clip.transition_duration_seconds for clip in timeline.clips[1:]),
        3,
    )


def create_timeline_revision(
    folder: Path,
    edits: list[TimelineIntervalEdit],
    *,
    timeline_path: Path | None = None,
    minimum_clip_duration_seconds: float = 1.0,
) -> TimelineRevisionResult:
    """Apply explicit trims to a separate, unapproved copy of an approved timeline.

    Edits may only narrow each clip's existing source interval. Extending source
    handles and changing clip order or media assignments are intentionally excluded.
    """
    if not edits:
        raise TimelineRevisionError("at least one --set-interval edit is required")
    if minimum_clip_duration_seconds <= 0:
        raise TimelineRevisionError("minimum clip duration must be positive")

    root = folder.expanduser().resolve()
    default_timeline = root / ".video-agent" / "optimized-timeline.json"
    source_file = (timeline_path or default_timeline).expanduser().resolve()
    try:
        source_file.relative_to(root)
    except ValueError as exc:
        raise TimelineRevisionError("source timeline must be inside the project folder") from exc
    if not source_file.is_file():
        raise FileNotFoundError(f"Missing source timeline: {source_file}")
    if not validate_artifact(root, source_file):
        raise TimelineRevisionError("source timeline is not current; regenerate it before revision")

    source = ProjectDraftTimeline.model_validate_json(source_file.read_text(encoding="utf-8"))
    if Path(source.project_path).resolve() != root:
        raise TimelineRevisionError("source timeline belongs to a different project folder")
    if not source.timeline_approved or any(clip.decision != "approved" for clip in source.clips):
        raise TimelineRevisionError("only a fully approved timeline can be revised")

    edit_map: dict[str, TimelineIntervalEdit] = {}
    for edit in edits:
        if edit.clip_id in edit_map:
            raise TimelineRevisionError(f"duplicate edit for clip {edit.clip_id}")
        edit_map[edit.clip_id] = edit

    clips_by_id = {clip.clip_id: clip for clip in source.clips}
    unknown = set(edit_map) - clips_by_id.keys()
    if unknown:
        raise TimelineRevisionError(f"unknown clip ID: {sorted(unknown)[0]}")

    revised_clips = [clip.model_copy(deep=True) for clip in source.clips]
    for clip in revised_clips:
        edit = edit_map.get(clip.clip_id)
        if edit is None:
            continue
        if (
            edit.source_start_seconds < clip.source_start_seconds - 0.001
            or edit.source_end_seconds > clip.source_end_seconds + 0.001
        ):
            raise TimelineRevisionError(
                f"edit for {clip.clip_id} must stay within its existing source interval "
                f"({clip.source_start_seconds:.3f}–{clip.source_end_seconds:.3f}s)"
            )
        duration = edit.source_end_seconds - edit.source_start_seconds
        if duration + 0.001 < minimum_clip_duration_seconds:
            raise TimelineRevisionError(
                f"edit for {clip.clip_id} is shorter than the "
                f"{minimum_clip_duration_seconds:.2f}s minimum"
            )
        if duration >= clip.duration_seconds - 0.001:
            raise TimelineRevisionError(
                f"edit for {clip.clip_id} must shorten the existing source interval"
            )
        clip.source_start_seconds = round(edit.source_start_seconds, 3)
        clip.source_end_seconds = round(edit.source_end_seconds, 3)
        clip.duration_seconds = round(duration, 3)
        clip.decision = "pending"
        clip.review_required = True
        if clip.pacing_target_seconds:
            ratio = clip.duration_seconds / clip.pacing_target_seconds
            clip.pacing_fit = "short" if ratio < 0.75 else "long" if ratio > 1.25 else "within"
            clip.pacing_note = (
                f"Revised shot is {clip.duration_seconds:.1f}s against a "
                f"{clip.pacing_target_seconds:.1f}s pacing target; review the new timing."
            )
    if revised_clips:
        revised_clips[0].transition_in = "cut"
        revised_clips[0].transition_duration_seconds = 0
    for index, clip in enumerate(revised_clips[1:], start=1):
        if clip.transition_in == "cut":
            continue
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

    revised_duration = round(
        sum(clip.duration_seconds for clip in revised_clips)
        - sum(clip.transition_duration_seconds for clip in revised_clips[1:]),
        3,
    )
    approved = [clip for clip in revised_clips if clip.decision == "approved"]
    approved_duration = round(
        sum(clip.duration_seconds for clip in approved)
        - sum(
            clip.transition_duration_seconds
            for previous, clip in zip(approved, approved[1:], strict=False)
            if clip.sequence_index == previous.sequence_index + 1
        ),
        3,
    )
    revised = ProjectDraftTimeline.model_validate(
        source.model_copy(
            update={
                "clips": revised_clips,
                "proposed_duration_seconds": revised_duration,
                "approved_duration_seconds": approved_duration,
                "timeline_approved": False,
            }
        ).model_dump()
    )

    revision_directory = root / ".video-agent" / "timeline-revisions"
    revision_directory.mkdir(parents=True, exist_ok=True)
    revision_number = 1
    while (revision_directory / f"revision-{revision_number:03}.json").exists():
        revision_number += 1
    revision_file = revision_directory / f"revision-{revision_number:03}.json"
    review_file = revision_file.with_suffix(".review.json")
    result = TimelineRevisionResult(
        source_timeline_path=source_file.relative_to(root).as_posix(),
        revision_path=revision_file.relative_to(root).as_posix(),
        changed_clip_ids=sorted(edit_map),
        before_duration_seconds=_timeline_duration(source),
        after_duration_seconds=revised_duration,
        timeline=revised,
    )
    for destination, payload in (\n        (revision_file, revised.model_dump(mode="json")),
        (review_file, result.model_dump(mode="json")),\n    ):\n        write_json_atomic(destination, payload)
    register_artifact(root, revision_file, dependencies=[source_file])
    return result


def compare_timeline_files(folder: Path, before_path: Path, after_path: Path) -> TimelineComparison:
    """Compare two project-local timeline files without changing either artifact."""
    root = folder.expanduser().resolve()
    timelines: list[tuple[Path, ProjectDraftTimeline]] = []
    for path, label in ((before_path, "before"), (after_path, "after")):
        candidate = path.expanduser()
        resolved = (
            (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        )
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise TimelineRevisionError(
                f"{label} timeline must be inside the project folder"
            ) from exc
        if not resolved.is_file():
            raise FileNotFoundError(f"Missing {label} timeline: {resolved}")
        timeline = ProjectDraftTimeline.model_validate_json(resolved.read_text(encoding="utf-8"))
        if Path(timeline.project_path).resolve() != root:
            raise TimelineRevisionError(f"{label} timeline belongs to a different project folder")
        timelines.append((resolved, timeline))

    before_file, before = timelines[0]
    after_file, after = timelines[1]
    before_clips = {clip.clip_id: clip for clip in before.clips}
    after_clips = {clip.clip_id: clip for clip in after.clips}
    changes: list[TimelineClipChange] = []
    for clip_id in sorted(before_clips.keys() | after_clips.keys()):
        old = before_clips.get(clip_id)
        new = after_clips.get(clip_id)
        if old is None or new is None:
            continue
        if (
            abs(old.source_start_seconds - new.source_start_seconds) > 0.001
            or abs(old.source_end_seconds - new.source_end_seconds) > 0.001
            or old.transition_in != new.transition_in
            or abs(old.transition_duration_seconds - new.transition_duration_seconds) > 0.001
        ):
            changes.append(
                TimelineClipChange(
                    clip_id=clip_id,
                    before_start_seconds=old.source_start_seconds,
                    after_start_seconds=new.source_start_seconds,
                    before_end_seconds=old.source_end_seconds,
                    after_end_seconds=new.source_end_seconds,
                    before_duration_seconds=old.duration_seconds,
                    after_duration_seconds=new.duration_seconds,
                    before_transition=(
                        f"{old.transition_in} ({old.transition_duration_seconds:.3f}s)"
                    ),
                    after_transition=(
                        f"{new.transition_in} ({new.transition_duration_seconds:.3f}s)"
                    ),
                )
            )
    before_duration = _timeline_duration(before)
    after_duration = _timeline_duration(after)
    return TimelineComparison(
        before_path=before_file.relative_to(root).as_posix(),
        after_path=after_file.relative_to(root).as_posix(),
        before_duration_seconds=before_duration,
        after_duration_seconds=after_duration,
        duration_delta_seconds=round(after_duration - before_duration, 3),
        changes=changes,
    )
