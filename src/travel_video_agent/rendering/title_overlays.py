"""Declarative, reviewable title overlays for approved draft timelines."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..artifact_tracking import register_artifact, validate_artifact
from ..editing.planning_schemas import VideoProjectPlan
from ..editing.timeline_schemas import ProjectDraftTimeline


class TitleOverlay(BaseModel):
    """A safe title instruction; styling maps to built-in renderer presets only."""

    model_config = ConfigDict(extra="forbid")

    overlay_id: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=120)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    placement: Literal["top", "center", "bottom"] = "bottom"
    style: Literal["cinematic", "minimal", "bold"] = "cinematic"
    animation: Literal["none", "fade", "slide_up"] = "fade"
    sequence_role: Literal["standard", "opening", "closing"] = "standard"
    decision: Literal["pending", "approved", "rejected"] = "pending"

    @model_validator(mode="after")
    def validate_interval(self) -> "TitleOverlay":
        if not self.text.strip() or any(ord(character) < 32 for character in self.text):
            raise ValueError("title text must be non-empty and contain no control characters")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("title end_seconds must be greater than start_seconds")
        if self.end_seconds - self.start_seconds < 0.5:
            raise ValueError("title overlays must remain visible for at least 0.5 seconds")
        return self


class TitleOverlayDocument(BaseModel):
    """Review artifact associated with one optimized timeline."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    project_path: str
    source_timeline: str
    overlays: list[TitleOverlay] = Field(default_factory=list)
    review_required: Literal[True] = True

    @model_validator(mode="after")
    def validate_overlays(self) -> "TitleOverlayDocument":
        ids = [item.overlay_id for item in self.overlays]
        if len(ids) != len(set(ids)):
            raise ValueError("title overlay IDs must be unique")
        return self


def load_title_overlays(
    folder: Path,
    overlay_path: Path | None = None,
    *,
    validate_overlay_artifact: bool = True,
) -> tuple[TitleOverlayDocument, Path]:
    """Load a project-local title document and validate its timeline dependency."""
    root = folder.expanduser().resolve()
    destination = (
        (overlay_path or root / ".video-agent" / "title-overlays.json").expanduser().resolve()
    )
    if not destination.is_file():
        raise FileNotFoundError(
            f"Missing {destination}; create .video-agent/title-overlays.json first."
        )
    if validate_overlay_artifact:
        validate_artifact(root, destination)
    document = TitleOverlayDocument.model_validate_json(destination.read_text(encoding="utf-8"))
    if Path(document.project_path).resolve() != root:
        raise ValueError("title overlay document belongs to a different project folder")
    timeline = Path(document.source_timeline)
    if not timeline.is_absolute():
        timeline = root / timeline
    timeline = timeline.resolve()
    try:
        timeline.relative_to(root)
    except ValueError as exc:
        raise ValueError("title overlay source timeline must be inside the project folder") from exc
    if not timeline.is_file():
        raise FileNotFoundError(f"Title overlay source timeline is missing: {timeline}")
    validate_artifact(root, timeline)
    ProjectDraftTimeline.model_validate_json(timeline.read_text(encoding="utf-8"))
    return document, destination


def review_title_overlays(
    document: TitleOverlayDocument,
    approve_ids: set[str],
    reject_ids: set[str],
) -> TitleOverlayDocument:
    """Apply explicit review decisions without modifying the source timeline."""
    if approve_ids & reject_ids:
        raise ValueError("a title overlay cannot be both approved and rejected")
    overlays = {item.overlay_id: item for item in document.overlays}
    if len(overlays) != len(document.overlays):
        raise ValueError("title overlay IDs must be unique")
    unknown = (approve_ids | reject_ids) - overlays.keys()
    if unknown:
        raise ValueError(f"unknown title overlay ID(s): {', '.join(sorted(unknown))}")
    reviewed = document.model_copy(deep=True)
    for item in reviewed.overlays:
        if item.overlay_id in approve_ids:
            item.decision = "approved"
        elif item.overlay_id in reject_ids:
            item.decision = "rejected"
    return TitleOverlayDocument.model_validate(reviewed.model_dump())


def save_title_overlays(
    folder: Path,
    document: TitleOverlayDocument,
    destination: Path,
    *,
    dependencies: list[Path] | None = None,
) -> None:
    """Persist reviewed overlays and track their timeline dependency."""
    root = folder.expanduser().resolve()
    destination = destination.expanduser().resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("title overlay output must be inside the project folder") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document.model_dump(mode="json"), indent=2) + chr(10), encoding="utf-8"
    )
    timeline = Path(document.source_timeline)
    if not timeline.is_absolute():
        timeline = root / timeline
    upstream = [timeline, *(dependencies or [])]
    register_artifact(root, destination, dependencies=upstream)



def suggest_narrative_title_sequences(
    folder: Path,
    *,
    timeline_path: Path | None = None,
    plan_path: Path | None = None,
    output_path: Path | None = None,
) -> tuple[TitleOverlayDocument, Path]:
    """Create review-only opening and closing titles from the creative story treatment."""
    root = folder.expanduser().resolve()
    timeline_file = (
        timeline_path or root / ".video-agent" / "optimized-timeline.json"
    ).expanduser().resolve()
    plan_file = (
        plan_path or root / ".video-agent" / "video-project-plan.json"
    ).expanduser().resolve()
    destination = (
        output_path or root / ".video-agent" / "title-overlays.json"
    ).expanduser().resolve()

    for artifact, label in ((timeline_file, "optimized timeline"), (plan_file, "video project plan")):
        if not artifact.is_file():
            raise FileNotFoundError(f"Missing {label}: {artifact}")
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{label} must be inside the project folder") from exc
        validate_artifact(root, artifact)

    if destination.exists():
        raise FileExistsError(
            f"Title overlay file already exists: {destination}; review or choose another output."
        )

    timeline = ProjectDraftTimeline.model_validate_json(
        timeline_file.read_text(encoding="utf-8")
    )
    plan = VideoProjectPlan.model_validate_json(plan_file.read_text(encoding="utf-8"))
    if Path(timeline.project_path).resolve() != root or Path(plan.project_path).resolve() != root:
        raise ValueError("title sequence inputs belong to a different project folder")
    if not timeline.timeline_approved:
        raise ValueError("the optimized timeline must be approved before suggesting title sequences")
    if plan.story_treatment is None:
        raise ValueError("the video project plan must contain a story treatment")
    approved = [clip for clip in timeline.clips if clip.decision == "approved"]
    if not approved:
        raise ValueError("the approved timeline contains no approved clips")

    duration = sum(clip.duration_seconds for clip in approved)
    duration -= sum(
        clip.transition_duration_seconds
        for previous, clip in zip(approved, approved[1:], strict=False)
        if clip.sequence_index == previous.sequence_index + 1
        and clip.transition_in != "cut"
    )
    if duration < 1.5:
        raise ValueError("the approved timeline is too short for opening and closing title sequences")

    sequence_duration = max(0.75, min(4.0, duration * 0.22))
    opening_end = sequence_duration
    closing_start = duration - sequence_duration
    if closing_start < opening_end:
        raise ValueError("the approved timeline is too short for separate opening and closing sequences")

    treatment = plan.story_treatment
    opening_subtitle = treatment.logline.strip()
    if len(opening_subtitle) > 120:
        opening_subtitle = opening_subtitle[:117].rstrip() + "..."
    closing_text = treatment.narrative_arc[-1].strip()
    if len(closing_text) > 120:
        closing_text = closing_text[:117].rstrip() + "..."
    if not closing_text:
        closing_text = "The journey continues"

    document = TitleOverlayDocument(
        project_path=str(root),
        source_timeline=str(timeline_file.relative_to(root)),
        overlays=[
            TitleOverlay(
                overlay_id="opening-title",
                text=treatment.working_title,
                start_seconds=0,
                end_seconds=opening_end,
                placement="center",
                style="cinematic",
                animation="fade",
                sequence_role="opening",
            ),
            TitleOverlay(
                overlay_id="opening-logline",
                text=opening_subtitle,
                start_seconds=0,
                end_seconds=opening_end,
                placement="bottom",
                style="minimal",
                animation="fade",
                sequence_role="opening",
            ),
            TitleOverlay(
                overlay_id="closing-card",
                text=closing_text,
                start_seconds=closing_start,
                end_seconds=duration,
                placement="center",
                style="cinematic",
                animation="slide_up",
                sequence_role="closing",
            ),
        ],
    )
    save_title_overlays(
        root, document, destination, dependencies=[plan_file]
    )
    return document, destination
