"""Translate rendered-draft critique evidence into reviewable shot-level suggestions."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..artifact_tracking import register_artifact, validate_artifact
from .timeline_schemas import ProjectDraftTimeline


class RevisionSuggestion(BaseModel):
    """A bounded recommendation tied to one existing timeline shot."""

    model_config = ConfigDict(extra="forbid")

    suggestion_id: str
    clip_id: str
    category: Literal["story", "pacing"]
    priority: Literal["high", "normal"]
    action: Literal["review_footage", "review_timing"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    timestamp_seconds: float | None = Field(default=None, ge=0)
    current_source_start_seconds: float = Field(ge=0)
    current_source_end_seconds: float = Field(gt=0)
    current_duration_seconds: float = Field(gt=0)
    summary: str
    evidence: str
    recommendation: str
    status: Literal["pending"] = "pending"


class DraftRevisionSuggestions(BaseModel):
    """Human-reviewable shot-level suggestions derived from a critique report."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    project_path: str
    source_critique: str
    source_timeline: str
    suggestions: list[RevisionSuggestion] = Field(default_factory=list)
    review_required: Literal[True] = True
    timeline_modified: Literal[False] = False
    footage_modified: Literal[False] = False


class RevisionSuggestionError(ValueError):
    """Raised when critique evidence cannot safely be mapped to a timeline."""


def build_draft_revision_suggestions(
    folder: Path,
    *,
    critique_path: Path | None = None,
    timeline_path: Path | None = None,
    minimum_semantic_confidence: float = 0.55,
) -> DraftRevisionSuggestions:
    """Map confident semantic concerns and pacing outliers to existing timeline shots.

    This stage deliberately proposes review actions rather than inventing source
    intervals. Exact trims, footage replacements, approvals, and rendering remain
    separate, explicit user decisions.
    """
    if not 0 <= minimum_semantic_confidence <= 1:
        raise ValueError("minimum_semantic_confidence must be between 0 and 1")
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project folder does not exist: {root}")
    critique_file = (critique_path or root / ".video-agent" / "draft-critique.json")
    timeline_file = (timeline_path or root / ".video-agent" / "optimized-timeline.json")
    critique_file = (
        (root / critique_file).resolve()
        if not critique_file.is_absolute()
        else critique_file.resolve()
    )
    timeline_file = (
        (root / timeline_file).resolve()
        if not timeline_file.is_absolute()
        else timeline_file.resolve()
    )
    for path, label in ((critique_file, "critique report"), (timeline_file, "timeline")):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RevisionSuggestionError(f"{label} must be inside the project folder") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
        if not validate_artifact(root, path):
            raise RevisionSuggestionError(
                f"{label} is not current; regenerate it before continuing"
            )

    try:
        critique = json.loads(critique_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RevisionSuggestionError("Critique report is not valid JSON") from exc
    if Path(critique.get("project_path", "")).resolve() != root:
        raise RevisionSuggestionError("Critique report belongs to a different project folder")
    if (root / critique.get("source_timeline", "")).resolve() != timeline_file:
        raise RevisionSuggestionError("Critique report references a different source timeline")

    timeline = ProjectDraftTimeline.model_validate_json(timeline_file.read_text(encoding="utf-8"))
    if Path(timeline.project_path).resolve() != root:
        raise RevisionSuggestionError("Timeline belongs to a different project folder")
    if not timeline.timeline_approved or any(
        clip.decision != "approved" for clip in timeline.clips
    ):
        raise RevisionSuggestionError("The source timeline must be fully reviewed and approved")
    clips = {clip.clip_id: clip for clip in timeline.clips}
    suggestions: list[RevisionSuggestion] = []
    seen: set[tuple[str, str]] = set()

    for review in critique.get("semantic_reviews", []):
        confidence = float(review.get("confidence", 0))
        match = review.get("match")
        shot_id = str(review.get("shot_id", ""))
        if match not in {"partial", "mismatch"} or confidence < minimum_semantic_confidence:
            continue
        clip = clips.get(shot_id)
        if clip is None:
            continue
        key = (clip.clip_id, "story")
        if key in seen:
            continue
        seen.add(key)
        evidence = str(review.get("evidence", "")).strip()
        suggestion = str(review.get("suggestion", "")).strip()
        suggestions.append(
            RevisionSuggestion(
                suggestion_id=f"{clip.clip_id}-story",
                clip_id=clip.clip_id,
                category="story",
                priority="high" if match == "mismatch" else "normal",
                action="review_footage",
                confidence=confidence,
                timestamp_seconds=float(review.get("timestamp_seconds", 0)),
                current_source_start_seconds=clip.source_start_seconds,
                current_source_end_seconds=clip.source_end_seconds,
                current_duration_seconds=clip.duration_seconds,
                summary=f"Storyboard match is {match} for shot {clip.clip_id}.",
                evidence=evidence or "The semantic review did not provide supporting detail.",
                recommendation=(
                    suggestion
                    or (
                        "Inspect the shot against its storyboard intent and consider a "
                        "different footage assignment."
                    )
                ),
            )
        )

    for clip in timeline.clips:
        if clip.decision != "approved" or clip.pacing_fit not in {"short", "long"}:
            continue
        key = (clip.clip_id, "pacing")
        if key in seen:
            continue
        seen.add(key)
        direction = "shorter" if clip.pacing_fit == "long" else "longer"
        suggestions.append(
            RevisionSuggestion(
                suggestion_id=f"{clip.clip_id}-pacing",
                clip_id=clip.clip_id,
                category="pacing",
                priority="normal",
                action="review_timing",
                current_source_start_seconds=clip.source_start_seconds,
                current_source_end_seconds=clip.source_end_seconds,
                current_duration_seconds=clip.duration_seconds,
                summary=f"Shot {clip.clip_id} is outside its pacing target.",
                evidence=(
                    f"Current duration {clip.duration_seconds:.2f}s; target "
                    f"{clip.pacing_target_seconds:.2f}s; recorded fit: {clip.pacing_fit}."
                ),
                recommendation=(
                    f"Review whether this shot should be {direction}. The current revision "
                    "tool supports safe trims only; do not extend beyond the reviewed "
                    "source interval."
                    if direction == "shorter"
                    else (
                        "Review whether the shot should remain as-is or use a different, "
                        "longer source selection. No interval extension is proposed."
                    )
                ),
            )
        )

    suggestions.sort(key=lambda item: (item.priority != "high", item.category, item.clip_id))
    return DraftRevisionSuggestions(
        project_path=str(root),
        source_critique=critique_file.relative_to(root).as_posix(),
        source_timeline=timeline_file.relative_to(root).as_posix(),
        suggestions=suggestions,
    )


def save_draft_revision_suggestions(
    folder: Path,
    report: DraftRevisionSuggestions,
    destination: Path | None = None,
) -> Path:
    """Persist suggestions with explicit dependencies on critique and source timeline."""
    root = folder.expanduser().resolve()
    output = destination or root / ".video-agent" / "draft-revision-suggestions.json"
    output = (root / output).resolve() if not output.is_absolute() else output.resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise RevisionSuggestionError(
            "Suggestion output must be inside the project folder"
        ) from exc
    if output in {root / report.source_critique, root / report.source_timeline}:
        raise RevisionSuggestionError("Suggestion output cannot overwrite an input artifact")
    if output.exists():
        raise FileExistsError(f"Suggestion output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    payload = json.dumps(report.model_dump(mode="json"), indent=2) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(output)
    register_artifact(
        root,
        output,
        dependencies=[root / report.source_critique, root / report.source_timeline],
    )
    return output
