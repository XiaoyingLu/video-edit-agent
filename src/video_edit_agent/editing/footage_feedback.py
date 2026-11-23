"""Project-local learning signals from explicit footage review decisions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..artifact_tracking import register_artifact, validate_artifact

FEEDBACK_FILENAME = "footage-feedback.json"


class FootageFeedbackEntry(BaseModel):
    """One explicit decision tied to a normalized storyboard-shot description."""

    model_config = ConfigDict(extra="forbid")

    shot_tokens: list[str] = Field(min_length=1, max_length=100)
    shot_scale: str = "unknown"
    visual_role: str = "unknown"
    selected_scene_id: str | None = None
    rejected_scene_ids: list[str] = Field(default_factory=list, max_length=100)


class ProjectFootageFeedback(BaseModel):
    """Small, local-only preference history; it never assigns footage."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    project_path: str
    entries: list[FootageFeedbackEntry] = Field(default_factory=list, max_length=5000)


def shot_feedback_tokens(shot) -> list[str]:
    """Return stable content tokens for matching similar future storyboard shots."""
    text = f"{shot.purpose} {shot.visual_description}".casefold()
    return sorted(set(re.findall(r"[a-z0-9]+", text)) - {"and", "for", "the", "with", "from"})


def _feedback_path(root: Path, path: Path | None = None) -> Path:
    destination = (path or root / ".video-agent" / FEEDBACK_FILENAME).expanduser()
    destination = (
        (root / destination).resolve() if not destination.is_absolute() else destination.resolve()
    )
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("footage feedback must be inside the project folder") from exc
    return destination


def load_footage_feedback(
    folder: Path, *, feedback_path: Path | None = None
) -> ProjectFootageFeedback:
    root = folder.expanduser().resolve()
    path = _feedback_path(root, feedback_path)
    if not path.exists():
        return ProjectFootageFeedback(project_path=str(root))
    if not path.is_file() or not validate_artifact(root, path):
        raise ValueError("footage feedback is not a current tracked project artifact")
    try:
        feedback = ProjectFootageFeedback.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError(f"could not load footage feedback: {exc}") from exc
    if Path(feedback.project_path).resolve() != root:
        raise ValueError("footage feedback belongs to a different project folder")
    return feedback


def record_footage_feedback(
    folder: Path,
    *,
    entries: list[FootageFeedbackEntry],
    feedback_path: Path | None = None,
) -> Path | None:
    """Append validated human decisions and register the resulting artifact."""
    if not entries:
        return None
    root = folder.expanduser().resolve()
    path = _feedback_path(root, feedback_path)
    feedback = load_footage_feedback(root, feedback_path=path)
    feedback.entries.extend(entries)
    if len(feedback.entries) > 5000:
        feedback.entries = feedback.entries[-5000:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(feedback.model_dump_json(indent=2) + "\n", encoding="utf-8")
    register_artifact(root, path)
    return path
