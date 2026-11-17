"""End-to-end natural-language video creation workflow."""

import json
from pathlib import Path

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

    plan = parse_editing_intent(prompt, model=vision_model)
    (artifact_dir / "editing-plan.json").write_text(
        json.dumps(plan.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )

    understanding = understand_folder(
        root,
        **({"model": vision_model} if vision_model else {}),
        samples_per_scene=samples_per_scene,
    )
    (artifact_dir / "understanding.json").write_text(
        json.dumps(understanding.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )

    transcription = transcribe_folder(root, model=whisper_model)
    (artifact_dir / "transcription.json").write_text(
        json.dumps(transcription.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )

    highlights = find_highlights_folder(root, samples_per_scene=samples_per_scene)
    (artifact_dir / "highlights.json").write_text(
        json.dumps(highlights.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    if not highlights.candidates:
        raise ValueError(
            "No highlight candidates were found; check that the folder contains readable videos."
        )

    selection = build_draft_selection(root)
    (artifact_dir / "draft-selection.json").write_text(
        json.dumps(selection.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    if selection.selected_clip_count == 0:
        raise ValueError("No footage could be selected for the requested edit.")

    timeline = build_draft_timeline(root)
    timeline_path = artifact_dir / "optimized-timeline.json"
    save_draft_timeline(timeline, timeline_path)
    if not timeline.clips:
        raise ValueError("The proposed shot timeline is empty.")
    return plan, highlights, selection, timeline, timeline_path
