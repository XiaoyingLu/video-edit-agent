"""Creation and persistence of hierarchical video project plans."""

import json
from pathlib import Path

from ..artifact_tracking import register_artifact
from .intent import parse_editing_intent
from .planning_schemas import CreativeBrief, VideoProjectPlan


def create_video_project_plan(
    folder: Path,
    prompt: str,
    *,
    model: str | None = None,
    output_path: Path | None = None,
) -> tuple[VideoProjectPlan, Path]:
    """Parse a creative brief and persist a planning artifact without analyzing or editing media."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project folder does not exist: {root}")

    editing_plan = parse_editing_intent(prompt, model=model)
    artifact_dir = root / ".video-agent"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    editing_path = artifact_dir / "editing-plan.json"
    editing_path.write_text(
        json.dumps(editing_plan.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    register_artifact(
        root,
        editing_path,
        config={"prompt": prompt, "model": model},
    )

    brief = CreativeBrief(
        source_prompt=editing_plan.source_prompt,
        target_duration_seconds=editing_plan.target_duration_seconds,
        style=editing_plan.style,
        aspect_ratio=editing_plan.aspect_ratio,
        resolution=editing_plan.resolution,
        pacing=editing_plan.pacing,
        creative_priorities=editing_plan.priorities,
        assumptions=editing_plan.assumptions,
    )
    project_plan = VideoProjectPlan(
        project_path=str(root),
        brief=brief,
        editing_preferences=editing_plan,
    )
    destination = (output_path or artifact_dir / "video-project-plan.json").expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(project_plan.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    register_artifact(root, destination, dependencies=[editing_path])
    return project_plan, destination
