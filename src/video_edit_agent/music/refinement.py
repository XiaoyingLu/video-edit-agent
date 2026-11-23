"""Human-reviewed, non-destructive soundtrack timing refinement."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..artifact_tracking import register_artifact, validate_artifact
from ..atomic_io import write_json_atomic
from ..editing.timeline_schemas import ProjectDraftTimeline
from ..media import probe_video
from .sync import MusicSyncError, MusicSyncSuggestions


class MusicSyncReviewDecision(BaseModel):
    """A user's decision for one suggested timeline boundary."""

    model_config = ConfigDict(extra="forbid")

    boundary_id: str = Field(min_length=1)
    status: Literal["accepted", "rejected", "skipped", "infeasible"]
    reason: str | None = None


class MusicSyncReviewResult(BaseModel):
    """Persistable provenance for one music-sync refinement review."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    source_timeline_path: str
    suggestions_path: str
    revision_path: str
    decisions: list[MusicSyncReviewDecision] = Field(default_factory=list)
    review_required: Literal[True] = True
    timeline: ProjectDraftTimeline | None = None


def apply_music_sync_review(
    folder: Path,
    *,
    suggestions_path: Path | None = None,
    timeline_path: Path | None = None,
    accept_ids: set[str],
    reject_ids: set[str],
    minimum_clip_duration_seconds: float = 1.0,
) -> MusicSyncReviewResult:
    """Apply accepted beat shifts in memory while preserving source safety."""
    if minimum_clip_duration_seconds <= 0:
        raise ValueError("minimum_clip_duration_seconds must be positive")
    root = folder.expanduser().resolve()
    timeline_file = (timeline_path or root / ".video-agent" / "optimized-timeline.json").resolve()
    suggestions_file = (
        suggestions_path or root / ".video-agent" / "music-sync-suggestions.json"
    ).resolve()
    for path, label in ((timeline_file, "timeline"), (suggestions_file, "suggestions")):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise MusicSyncError(f"{label} artifact must be inside the project folder") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} artifact: {path}")
        if not validate_artifact(root, path):
            raise MusicSyncError(f"{label} artifact is not tracked; regenerate it before review")
    timeline = ProjectDraftTimeline.model_validate_json(timeline_file.read_text(encoding="utf-8"))
    suggestions = MusicSyncSuggestions.model_validate_json(
        suggestions_file.read_text(encoding="utf-8")
    )
    if (
        Path(timeline.project_path).resolve() != root
        or Path(suggestions.project_path).resolve() != root
    ):
        raise MusicSyncError("timeline and suggestions must belong to the project folder")
    if (root / suggestions.timeline_path).resolve() != timeline_file:
        raise MusicSyncError("music sync suggestions reference a different timeline")
    if accept_ids & reject_ids:
        raise MusicSyncError("a boundary cannot be both accepted and rejected")

    clips = [clip.model_copy(deep=True) for clip in timeline.clips]
    source_durations: dict[str, float] = {}
    for clip in clips:
        source = (root / clip.video_path).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise MusicSyncError(
                f"source video for {clip.clip_id} escapes the project folder"
            ) from exc
        if not source.is_file():
            raise FileNotFoundError(f"Missing source video for {clip.clip_id}: {source}")
        source_durations.setdefault(clip.video_path, probe_video(source, root).duration_seconds)

    known_ids = {
        f"{left.clip_id}->{right.clip_id}" for left, right in zip(clips, clips[1:], strict=False)
    }
    unknown = (accept_ids | reject_ids) - known_ids
    if unknown:
        raise MusicSyncError(f"unknown music sync boundary: {sorted(unknown)[0]}")
    recommendations = {
        f"{item.previous_clip_id}->{item.next_clip_id}": item for item in suggestions.boundaries
    }
    decisions: list[MusicSyncReviewDecision] = []
    for index, (previous, following) in enumerate(zip(clips, clips[1:], strict=False)):
        boundary_id = f"{previous.clip_id}->{following.clip_id}"
        recommendation = recommendations.get(boundary_id)
        if boundary_id in reject_ids:
            decisions.append(MusicSyncReviewDecision(boundary_id=boundary_id, status="rejected"))
            continue
        if boundary_id not in accept_ids:
            decisions.append(MusicSyncReviewDecision(boundary_id=boundary_id, status="skipped"))
            continue
        if recommendation is None or recommendation.adjustment_seconds is None:
            decisions.append(
                MusicSyncReviewDecision(
                    boundary_id=boundary_id,
                    status="infeasible",
                    reason="No applicable beat-alignment adjustment.",
                )
            )
            continue
        shift = recommendation.adjustment_seconds
        updated_previous = previous.model_copy(
            update={
                "source_end_seconds": round(previous.source_end_seconds + shift, 3),
                "duration_seconds": round(previous.duration_seconds + shift, 3),
            }
        )
        updated_following = following.model_copy(
            update={
                "source_start_seconds": round(following.source_start_seconds + shift, 3),
                "duration_seconds": round(following.duration_seconds - shift, 3),
            }
        )
        safe = all(
            candidate.source_start_seconds >= 0
            and candidate.source_end_seconds <= source_durations[candidate.video_path] + 0.001
            and candidate.duration_seconds + 0.001 >= minimum_clip_duration_seconds
            for candidate in (updated_previous, updated_following)
        )
        if not safe:
            decisions.append(
                MusicSyncReviewDecision(
                    boundary_id=boundary_id,
                    status="infeasible",
                    reason="The adjustment exceeds source handles or minimum clip duration.",
                )
            )
            continue
        clips[index] = updated_previous
        clips[index + 1] = updated_following
        decisions.append(MusicSyncReviewDecision(boundary_id=boundary_id, status="accepted"))

    revised = timeline.model_copy(
        update={
            "clips": clips,
            "proposed_duration_seconds": round(
                sum(clip.duration_seconds for clip in clips)
                - sum(clip.transition_duration_seconds for clip in clips[1:]),
                3,
            ),
            "timeline_approved": False,
        }
    )
    revised = ProjectDraftTimeline.model_validate(revised.model_dump())
    revision_directory = root / ".video-agent" / "music-sync-revisions"
    revision_directory.mkdir(parents=True, exist_ok=True)
    revision_number = 1
    while (revision_directory / f"revision-{revision_number:03}.json").exists():
        revision_number += 1
    revision_file = revision_directory / f"revision-{revision_number:03}.json"
    review_file = revision_file.with_suffix(".review.json")
    result = MusicSyncReviewResult(
        source_timeline_path=timeline_file.relative_to(root).as_posix(),
        suggestions_path=suggestions_file.relative_to(root).as_posix(),
        revision_path=revision_file.relative_to(root).as_posix(),
        decisions=decisions,
        timeline=revised,
    )
    for destination, payload in (
        (revision_file, revised.model_dump(mode="json")),
        (review_file, result.model_dump(mode="json")),
    ):
        write_json_atomic(destination, payload)
    register_artifact(root, revision_file, dependencies=[timeline_file, suggestions_file])
    return result
