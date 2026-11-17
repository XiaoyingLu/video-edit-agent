"""Creation and persistence of hierarchical video project plans."""

import json
from pathlib import Path

from ..artifact_tracking import register_artifact
from .intent import parse_editing_intent
from .planning_generation import (
    explicit_duration_seconds,
    generate_video_plan_content,
    infer_creative_priorities,
)
from .planning_schemas import CreativeBrief, VideoProjectPlan


def create_video_project_plan(
    folder: Path,
    prompt: str,
    *,
    model: str | None = None,
    output_path: Path | None = None,
) -> tuple[VideoProjectPlan, Path]:
    """Generate a creative brief, narrative treatment, and storyboard without editing media."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project folder does not exist: {root}")

    artifact_dir = root / ".video-agent"
    editing_path = artifact_dir / "editing-plan.json"
    destination = (output_path or artifact_dir / "video-project-plan.json").expanduser().resolve()
    if destination == editing_path.resolve():
        raise ValueError("video project plan output cannot overwrite editing-plan.json")

    editing_plan = parse_editing_intent(prompt, model=model)
    explicit_duration = explicit_duration_seconds(prompt)
    updates = {"priorities": infer_creative_priorities(prompt, editing_plan.priorities)}
    if explicit_duration is not None:
        if explicit_duration > 3600:
            raise ValueError("requested video duration cannot exceed 3600 seconds")
        updates["target_duration_seconds"] = explicit_duration
    editing_plan = editing_plan.model_copy(update=updates)

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
    generated = generate_video_plan_content(
        prompt, brief.model_dump(mode="json"), model=model
    )
    project_plan = VideoProjectPlan(
        project_path=str(root),
        brief=brief,
        editing_preferences=editing_plan,
        story_treatment=generated.story_treatment,
        scenes=generated.scenes,
        planning_stage="storyboard_ready",
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    editing_path.write_text(
        json.dumps(editing_plan.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    register_artifact(root, editing_path, config={"prompt": prompt, "model": model})

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(project_plan.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    register_artifact(root, destination, dependencies=[editing_path])
    return project_plan, destination
