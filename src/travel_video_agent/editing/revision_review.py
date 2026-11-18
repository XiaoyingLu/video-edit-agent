"""Human review and optional safe trims for critique-driven suggestions."""

import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..artifact_tracking import register_artifact, validate_artifact
from .revisions import (
    TimelineIntervalEdit,
    TimelineRevisionResult,
    create_timeline_revision,
)
from .timeline_schemas import ProjectDraftTimeline


class RevisionSuggestionDecision(BaseModel):
    """Explicit human decision for one suggestion."""

    model_config = ConfigDict(extra="forbid")

    suggestion_id: str
    clip_id: str
    status: Literal["accepted", "rejected", "skipped"]
    note: str | None = None


class RevisionSuggestionReviewResult(BaseModel):
    """Persisted review and optional separate timeline revision."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    suggestions_path: str
    source_timeline_path: str
    decisions: list[RevisionSuggestionDecision] = Field(default_factory=list)
    accepted_footage_reviews: list[str] = Field(default_factory=list)
    accepted_timing_reviews: list[str] = Field(default_factory=list)
    timeline_revision_path: str | None = None
    source_timeline_modified: Literal[False] = False
    timeline_revision_created: bool = False
    review_required: Literal[True] = True


class RevisionSuggestionReviewError(ValueError):
    """Raised when suggestion decisions cannot be applied safely."""


def review_draft_revision_suggestions(
    folder: Path,
    *,
    suggestions_path: Path | None = None,
    accept_ids: set[str],
    reject_ids: set[str],
    interval_edits: list[TimelineIntervalEdit] | None = None,
) -> tuple[RevisionSuggestionReviewResult, Path, TimelineRevisionResult | None]:
    """Record explicit decisions and optionally create a separate trim revision."""
    root = folder.expanduser().resolve()
    suggestion_file = (
        suggestions_path or root / ".video-agent" / "draft-revision-suggestions.json"
    ).expanduser()
    suggestion_file = (
        (root / suggestion_file).resolve()
        if not suggestion_file.is_absolute()
        else suggestion_file.resolve()
    )
    try:
        suggestion_file.relative_to(root)
    except ValueError as exc:
        raise RevisionSuggestionReviewError(
            "Suggestion artifact must be inside the project folder"
        ) from exc
    if not suggestion_file.is_file():
        raise FileNotFoundError(f"Missing suggestion artifact: {suggestion_file}")
    if not validate_artifact(root, suggestion_file):
        raise RevisionSuggestionReviewError(
            "Suggestion artifact is not current; regenerate it before review"
        )

    try:
        payload = json.loads(suggestion_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RevisionSuggestionReviewError("Suggestion artifact is not valid JSON") from exc
    suggestions = payload.get("suggestions", [])
    known = {str(item.get("suggestion_id")): item for item in suggestions}
    if accept_ids & reject_ids:
        raise RevisionSuggestionReviewError("A suggestion cannot be both accepted and rejected")
    unknown = (accept_ids | reject_ids) - known.keys()
    if unknown:
        raise RevisionSuggestionReviewError(f"Unknown suggestion ID: {sorted(unknown)[0]}")
    if not suggestions:
        raise RevisionSuggestionReviewError("There are no suggestions to review")

    timeline_file = (root / payload.get("source_timeline", "")).resolve()
    try:
        timeline_file.relative_to(root)
    except ValueError as exc:
        raise RevisionSuggestionReviewError(
            "Source timeline path in the suggestion artifact is invalid"
        ) from exc
    if not timeline_file.is_file() or not validate_artifact(root, timeline_file):
        raise RevisionSuggestionReviewError(
            "Source timeline is missing or stale; regenerate suggestions before review"
        )
    timeline = ProjectDraftTimeline.model_validate_json(
        timeline_file.read_text(encoding="utf-8")
    )
    if Path(payload.get("project_path", "")).resolve() != root:
        raise RevisionSuggestionReviewError("Suggestions belong to a different project folder")
    if Path(timeline.project_path).resolve() != root:
        raise RevisionSuggestionReviewError("Timeline belongs to a different project folder")
    if not timeline.timeline_approved or any(
        clip.decision != "approved" for clip in timeline.clips
    ):
        raise RevisionSuggestionReviewError("Only a fully approved source timeline can be reviewed")

    decisions: list[RevisionSuggestionDecision] = []
    accepted_footage: list[str] = []
    accepted_timing: list[str] = []
    for suggestion_id, item in known.items():
        status = (
            "accepted"
            if suggestion_id in accept_ids
            else "rejected"
            if suggestion_id in reject_ids
            else "skipped"
        )
        decisions.append(
            RevisionSuggestionDecision(
                suggestion_id=suggestion_id,
                clip_id=str(item["clip_id"]),
                status=status,
            )
        )
        if status == "accepted":
            if item.get("action") == "review_footage":
                accepted_footage.append(suggestion_id)
            elif item.get("action") == "review_timing":
                accepted_timing.append(suggestion_id)

    edits = interval_edits or []
    accepted_timing_clips = {
        str(known[suggestion_id]["clip_id"]) for suggestion_id in accepted_timing
    }
    if len({edit.clip_id for edit in edits}) != len(edits):
        raise RevisionSuggestionReviewError("Only one interval edit per shot is allowed")
    unauthorized = {edit.clip_id for edit in edits} - accepted_timing_clips
    if unauthorized:
        raise RevisionSuggestionReviewError(
            f"Interval edit requires an accepted timing suggestion: {sorted(unauthorized)[0]}"
        )
    clip_map = {clip.clip_id: clip for clip in timeline.clips}
    for edit in edits:
        clip = clip_map.get(edit.clip_id)
        if clip is None or clip.pacing_fit != "long":
            raise RevisionSuggestionReviewError(
                "Only accepted overlong shots can be trimmed in this workflow; "
                "short shots require a separate footage-selection decision"
            )

    revision = (
        create_timeline_revision(root, edits, timeline_path=timeline_file) if edits else None
    )
    result = RevisionSuggestionReviewResult(
        suggestions_path=suggestion_file.relative_to(root).as_posix(),
        source_timeline_path=timeline_file.relative_to(root).as_posix(),
        decisions=decisions,
        accepted_footage_reviews=accepted_footage,
        accepted_timing_reviews=accepted_timing,
        timeline_revision_path=revision.revision_path if revision else None,
        timeline_revision_created=revision is not None,
    )

    review_directory = root / ".video-agent" / "revision-reviews"
    review_directory.mkdir(parents=True, exist_ok=True)
    review_number = 1
    while (review_directory / f"review-{review_number:03}.json").exists():
        review_number += 1
    review_file = review_directory / f"review-{review_number:03}.json"
    temporary = review_file.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, review_file)
    dependencies = [suggestion_file, timeline_file]
    if revision is not None:
        dependencies.append(root / revision.revision_path)
    register_artifact(root, review_file, dependencies=dependencies)
    return result, review_file, revision
