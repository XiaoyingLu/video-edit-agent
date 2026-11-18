"""Declarative, reviewable title overlays for approved draft timelines."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..artifact_tracking import register_artifact, validate_artifact
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
        overlay_path or root / ".video-agent" / "title-overlays.json"
    ).expanduser().resolve()
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
    folder: Path, document: TitleOverlayDocument, destination: Path
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
    register_artifact(root, destination, dependencies=[timeline])
