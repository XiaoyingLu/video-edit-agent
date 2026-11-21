"""End-to-end natural-language video creation workflow."""

import json
import os
from pathlib import Path

from ..artifact_tracking import register_artifact
from ..highlights import find_highlights_folder
from ..highlights.schemas import ProjectHighlights
from ..transcription import transcribe_folder
from ..understanding import understand_folder
from .draft_pipeline import build_draft_selection
from .draft_schemas import ProjectDraftSelection
from .intent import parse_editing_intent
from .intent_schemas import EditingPlan
from .timeline_pipeline import build_draft_timeline, save_draft_timeline
from .timeline_schemas import ProjectDraftTimeline


def _save_artifact(
    root: Path,
    name: str,
    payload: object,
    *,
    dependencies: list[str] | None = None,
    config: dict | None = None,
) -> Path:
    """Persist JSON and register its source/dependency provenance."""
    destination = root / ".video-agent" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    register_artifact(root, destination, dependencies=dependencies, config=config)
    return destination


def create_video_draft(
    folder: Path,
    prompt: str,
    *,
    vision_model: str | None = None,
    whisper_model: str = "small",
    samples_per_scene: int = 2,
) -> tuple[EditingPlan, ProjectHighlights, ProjectDraftSelection, ProjectDraftTimeline, Path]:
    """Run analysis and deterministic planning stages, stopping before review/render."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project folder does not exist: {root}")
    artifact_dir = root / ".video-agent"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    resolved_vision_model = vision_model or os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
    plan = parse_editing_intent(prompt, model=resolved_vision_model)
    _save_artifact(
        root,
        "editing-plan.json",
        plan,
        config={"prompt": prompt, "model": resolved_vision_model},
    )

    understanding = understand_folder(
        root,
        model=resolved_vision_model,
        samples_per_scene=samples_per_scene,
    )
    _save_artifact(
        root,
        "understanding.json",
        understanding,
        config={
            "model": getattr(understanding, "model", resolved_vision_model),
            "samples_per_scene": samples_per_scene,
        },
    )

    transcription = transcribe_folder(root, model=whisper_model)
    _save_artifact(
        root,
        "transcription.json",
        transcription,
        config={"model": whisper_model},
    )

    highlights = find_highlights_folder(root, samples_per_scene=samples_per_scene)
    _save_artifact(
        root,
        "highlights.json",
        highlights,
        dependencies=[".video-agent/understanding.json"],
        config={"samples_per_scene": samples_per_scene},
    )
    if not highlights.candidates:
        raise ValueError(
            "No highlight candidates were found; check that the folder contains readable videos."
        )

    selection = build_draft_selection(root)
    _save_artifact(
        root,
        "draft-selection.json",
        selection,
        dependencies=[
            ".video-agent/editing-plan.json",
            ".video-agent/highlights.json",
            ".video-agent/understanding.json",
            ".video-agent/transcription.json",
        ],
    )
    if selection.selected_clip_count == 0:
        raise ValueError("No footage could be selected for the requested edit.")

    timeline = build_draft_timeline(root)
    timeline_path = artifact_dir / "optimized-timeline.json"
    save_draft_timeline(timeline, timeline_path)
    register_artifact(
        root,
        timeline_path,
        dependencies=[".video-agent/draft-selection.json", ".video-agent/editing-plan.json"],
    )
    if not timeline.clips:
        raise ValueError("The proposed shot timeline is empty.")
    return plan, highlights, selection, timeline, timeline_path
