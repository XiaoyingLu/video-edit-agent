"""Declarative, reviewable title overlays for approved draft timelines."""

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..artifact_tracking import register_artifact, validate_artifact
from ..atomic_io import write_json_atomic
from ..editing.planning_schemas import VideoProjectPlan
from ..editing.timeline_schemas import ProjectDraftTimeline
from ..templates import load_template


class TitleOverlay(BaseModel):
    """A safe title instruction; styling maps to built-in renderer presets only."""

    model_config = ConfigDict(extra="forbid")

    overlay_id: str = Field(min_length=1, max_length=80)
    text: str | None = Field(default=None, min_length=1, max_length=120)
    overlay_kind: Literal["title", "lower_third", "asset"] = "title"
    asset_path: str | None = Field(default=None, max_length=240)
    asset_format: Literal["webm", "png_sequence"] | None = None
    asset_x: float = Field(default=0.5, ge=0, le=1)
    asset_y: float = Field(default=0.5, ge=0, le=1)
    asset_scale: float = Field(default=0.2, gt=0, le=1)
    asset_fps: int = Field(default=30, ge=1, le=120)
    secondary_text: str | None = Field(default=None, max_length=100)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    placement: Literal["top", "center", "bottom"] = "bottom"
    style: Literal["cinematic", "minimal", "bold"] = "cinematic"
    animation: Literal["none", "fade", "slide_up"] = "fade"
    easing: Literal["linear", "ease_in", "ease_out", "ease_in_out"] = "linear"
    exit_animation: Literal["auto", "none", "fade", "slide_down", "slide_up"] = "auto"
    sequence_role: Literal["standard", "opening", "closing"] = "standard"
    decision: Literal["pending", "approved", "rejected"] = "pending"

    @model_validator(mode="after")
    def validate_interval(self) -> "TitleOverlay":
        if self.overlay_kind == "asset":
            if self.text is not None or self.secondary_text is not None:
                raise ValueError("asset overlays cannot contain title text")
            if not self.asset_path or not self.asset_format:
                raise ValueError("asset overlays require asset_path and asset_format")
            if Path(self.asset_path).is_absolute() or "\\" in self.asset_path:
                raise ValueError("asset_path must be a project-relative POSIX path")
            if not self.asset_path.startswith("templates/overlays/"):
                raise ValueError("asset_path must be inside templates/overlays/")
            if self.asset_format == "webm" and not self.asset_path.lower().endswith(".webm"):
                raise ValueError("webm assets must use a .webm file")
            if self.asset_format == "png_sequence" and not (
                self.asset_path.lower().endswith(".png") and re.search(r"%0\d+d", self.asset_path)
            ):
                raise ValueError(
                    "PNG sequences must use a numbered .png pattern such as frame-%04d.png"
                )
        else:
            if (
                self.text is None
                or not self.text.strip()
                or any(ord(character) < 32 for character in self.text)
            ):
                raise ValueError("title text must be non-empty and contain no control characters")
            if self.asset_path is not None or self.asset_format is not None:
                raise ValueError(
                    "asset_path and asset_format are only supported for asset overlays"
                )
        if self.secondary_text is not None and (
            not self.secondary_text.strip()
            or any(ord(character) < 32 for character in self.secondary_text)
        ):
            raise ValueError("secondary text must be non-empty and contain no control characters")
        if self.overlay_kind != "lower_third" and self.secondary_text is not None:
            raise ValueError("secondary text is only supported for lower-third overlays")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("title end_seconds must be greater than start_seconds")
        if self.end_seconds - self.start_seconds < 0.5:
            raise ValueError("overlays must remain visible for at least 0.5 seconds")
        return self


class TitleOverlayDocument(BaseModel):
    """Review artifact associated with one optimized timeline."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0", "1.1"] = "1.0"
    project_path: str
    source_timeline: str
    source_plan: str | None = None
    template_id: str | None = None
    template_version: int | None = Field(default=None, ge=1)
    template_source: str | None = None
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
    if document.source_plan is not None:
        plan = Path(document.source_plan)
        if not plan.is_absolute():
            plan = root / plan
        plan = plan.resolve()
        try:
            plan.relative_to(root)
        except ValueError as exc:
            raise ValueError("title overlay source plan must be inside the project folder") from exc
        if not plan.is_file():
            raise FileNotFoundError(f"Title overlay source plan is missing: {plan}")
        validate_artifact(root, plan)
        VideoProjectPlan.model_validate_json(plan.read_text(encoding="utf-8"))
    return document, destination


def resolve_asset_overlays(
    folder: Path, overlays: list[TitleOverlay]
) -> list[tuple[TitleOverlay, Path]]:
    """Resolve approved asset files, enforcing the project template-folder boundary."""
    root = folder.expanduser().resolve()
    resolved: list[tuple[TitleOverlay, Path]] = []
    allowed = (root / "templates" / "overlays").resolve()
    try:
        allowed.relative_to(root)
    except ValueError as exc:
        raise ValueError("asset overlay template folder must be inside the project folder") from exc
    for overlay in overlays:
        if overlay.overlay_kind != "asset" or overlay.decision != "approved":
            continue
        if overlay.asset_path is None or overlay.asset_format is None:
            raise ValueError(f"asset overlay {overlay.overlay_id} is missing its asset reference")
        asset = (root / overlay.asset_path).resolve()
        try:
            asset.relative_to(allowed)
        except ValueError as exc:
            raise ValueError("asset overlay path must remain inside templates/overlays") from exc
        if overlay.asset_format == "webm":
            if not asset.is_file():
                raise FileNotFoundError(f"asset overlay file does not exist: {asset}")
        else:
            pattern = re.sub(r"%0\d+d", "*", asset.name)
            frames = list(asset.parent.glob(pattern))
            if not frames:
                raise FileNotFoundError(f"PNG sequence has no matching frames: {asset}")
            for frame in frames:
                try:
                    frame.resolve().relative_to(allowed)
                except ValueError as exc:
                    raise ValueError("PNG sequence frame escapes templates/overlays") from exc
        resolved.append((overlay, asset))
    return resolved


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
    write_json_atomic(destination, document)
    timeline = Path(document.source_timeline)
    if not timeline.is_absolute():
        timeline = root / timeline
    upstream = [timeline, *(dependencies or [])]
    if document.source_plan is not None:
        plan = Path(document.source_plan)
        if not plan.is_absolute():
            plan = root / plan
        upstream.append(plan.resolve())
    if document.template_source is not None:
        upstream.append(Path(document.template_source).expanduser().resolve())
    register_artifact(root, destination, dependencies=upstream)


def _template_overlay_style(template, style_name: str) -> dict:
    """Translate a validated template style into the current safe overlay options."""
    style = template.title_styles[style_name]
    renderer_style = "minimal" if style.box == 0 else ("bold" if style.size >= 1.1 else "cinematic")
    return {
        "placement": style.placement,
        "style": renderer_style,
        "animation": style.animation if style.animation in {"none", "fade", "slide_up"} else "fade",
    }


def suggest_narrative_title_sequences(
    folder: Path,
    *,
    timeline_path: Path | None = None,
    plan_path: Path | None = None,
    output_path: Path | None = None,
    template_id: str = "cinematic",
) -> tuple[TitleOverlayDocument, Path]:
    """Create template-driven opening and closing titles for human review only."""
    root = folder.expanduser().resolve()
    timeline_file = (
        (timeline_path or root / ".video-agent" / "optimized-timeline.json").expanduser().resolve()
    )
    plan_file = (
        (plan_path or root / ".video-agent" / "video-project-plan.json").expanduser().resolve()
    )
    destination = (
        (output_path or root / ".video-agent" / "title-overlays.json").expanduser().resolve()
    )

    for artifact, label in (
        (timeline_file, "optimized timeline"),
        (plan_file, "video project plan"),
    ):
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

    template, template_path = load_template(template_id, root)
    timeline = ProjectDraftTimeline.model_validate_json(timeline_file.read_text(encoding="utf-8"))
    plan = VideoProjectPlan.model_validate_json(plan_file.read_text(encoding="utf-8"))
    if Path(timeline.project_path).resolve() != root or Path(plan.project_path).resolve() != root:
        raise ValueError("title sequence inputs belong to a different project folder")
    if not timeline.timeline_approved:
        raise ValueError(
            "the optimized timeline must be approved before suggesting title sequences"
        )
    if plan.story_treatment is None:
        raise ValueError("the video project plan must contain a story treatment")
    approved = [clip for clip in timeline.clips if clip.decision == "approved"]
    if not approved:
        raise ValueError("the approved timeline contains no approved clips")

    duration = sum(clip.duration_seconds for clip in approved)
    duration -= sum(
        clip.transition_duration_seconds
        for previous, clip in zip(approved, approved[1:], strict=False)
        if clip.sequence_index == previous.sequence_index + 1 and clip.transition_in != "cut"
    )
    if duration < 1.5:
        raise ValueError(
            "the approved timeline is too short for opening and closing title sequences"
        )

    sequence_duration = max(0.75, min(4.0, duration * 0.22))
    opening_end = sequence_duration
    closing_start = duration - sequence_duration
    if closing_start < opening_end:
        raise ValueError(
            "the approved timeline is too short for separate opening and closing sequences"
        )

    treatment = plan.story_treatment
    slot_values = {
        "{title}": treatment.working_title.strip(),
        "{logline}": treatment.logline.strip(),
        "{closing_line}": (treatment.narrative_arc[-1].strip() or "The journey continues"),
        "{date}": "",
        "{location}": "",
    }
    limits = {"{title}": 40, "{logline}": 48, "{closing_line}": 40, "{date}": 40, "{location}": 40}
    overlays: list[TitleOverlay] = []
    for sequence_name, start_seconds, end_seconds, role in (
        ("opening", 0.0, opening_end, "opening"),
        ("closing", closing_start, duration, "closing"),
    ):
        for index, item in enumerate(template.sequences.get(sequence_name, []), start=1):
            text_value = slot_values[item.slot]
            if not text_value:
                continue
            limit = limits[item.slot]
            if len(text_value) > limit:
                text_value = text_value[: limit - 3].rstrip() + "..."
            overlays.append(
                TitleOverlay(
                    overlay_id=f"{sequence_name}-{index}",
                    text=text_value,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    sequence_role=role,
                    **_template_overlay_style(template, item.style),
                )
            )

    if not overlays:
        raise ValueError(f"template {template_id!r} has no usable opening or closing title slots")

    document = TitleOverlayDocument(
        schema_version="1.1",
        project_path=str(root),
        source_timeline=str(timeline_file.relative_to(root)),
        source_plan=str(plan_file.relative_to(root)),
        template_id=template.id,
        template_version=template.version,
        template_source=str(template_path),
        overlays=overlays,
    )
    save_title_overlays(root, document, destination)
    return document, destination
