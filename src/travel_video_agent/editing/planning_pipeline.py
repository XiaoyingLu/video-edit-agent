"""Creation and persistence of hierarchical video project plans."""

import json
import os
from pathlib import Path

from ..artifact_tracking import register_artifact, validate_artifact
from ..music.schemas import MusicAnalysis, MusicSection
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
    music_analysis_path: Path | None = None,
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

    music_analysis_file = None
    music_analysis = None
    if music_analysis_path is not None:
        music_analysis_file = music_analysis_path.expanduser().resolve()
        try:
            music_analysis_file.relative_to(root)
        except ValueError as exc:
            raise ValueError("music analysis artifact must be inside the project folder") from exc
        if not music_analysis_file.is_file():
            raise FileNotFoundError(
                f"Music analysis artifact does not exist: {music_analysis_file}"
            )
        validate_artifact(root, music_analysis_file)
        music_analysis = MusicAnalysis.model_validate_json(
            music_analysis_file.read_text(encoding="utf-8")
        )
        if Path(music_analysis.project_path).expanduser().resolve() != root:
            raise ValueError("music analysis belongs to a different project folder")
        if not Path(music_analysis.music_path).expanduser().is_file():
            raise FileNotFoundError(
                f"Analyzed soundtrack no longer exists: {music_analysis.music_path}"
            )

    planner_model = model or os.getenv("VIDEO_PLANNER_MODEL", "qwen2.5:1.5b")
    editing_plan = parse_editing_intent(
        prompt,
        model=planner_model,
        memory_override_env="VIDEO_PLANNER_MIN_MEMORY_GB",
    )
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
    music_sections = (
        [section.model_dump(mode="json") for section in music_analysis.sections]
        if music_analysis is not None
        else []
    )
    generated = generate_video_plan_content(
        prompt,
        brief.model_dump(mode="json"),
        model=planner_model,
        music_sections=music_sections,
    )
    scenes = _annotate_scenes_with_music(generated.scenes, music_analysis.sections) if (
        music_analysis is not None
    ) else generated.scenes
    project_plan = VideoProjectPlan(
        project_path=str(root),
        brief=brief,
        editing_preferences=editing_plan,
        source_music_analysis=(
            music_analysis_file.relative_to(root).as_posix()
            if music_analysis_file is not None
            else None
        ),
        story_treatment=generated.story_treatment,
        scenes=scenes,
        planning_stage="storyboard_ready",
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    editing_path.write_text(
        json.dumps(editing_plan.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    register_artifact(root, editing_path, config={"prompt": prompt, "planner_model": planner_model})

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(project_plan.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    dependencies = [editing_path]
    if music_analysis_file is not None:
        dependencies.append(music_analysis_file)
    register_artifact(root, destination, dependencies=dependencies)
    return project_plan, destination


def _annotate_scenes_with_music(
    scenes: list,
    sections: list[MusicSection],
) -> list:
    """Attach the dominant overlapping soundtrack energy section to each planned scene."""
    ordered_sections = sorted(sections, key=lambda section: section.start_seconds)
    scene_start = 0.0
    annotated = []
    for scene in scenes:
        scene_end = scene_start + scene.target_duration_seconds
        overlaps = []
        for section in ordered_sections:
            overlap = min(scene_end, section.end_seconds) - max(
                scene_start, section.start_seconds
            )
            if overlap > 0:
                overlaps.append((overlap, section))
        if overlaps:
            overlap, dominant = max(
                overlaps,
                key=lambda item: (item[0], -item[1].start_seconds),
            )
            note = (
                f"Dominant {dominant.energy}-energy section overlaps "
                f"{overlap:.1f}s of this scene."
            )
            annotated.append(
                scene.model_copy(
                    update={
                        "music_energy": dominant.energy,
                        "music_section_start_seconds": dominant.start_seconds,
                        "music_section_end_seconds": dominant.end_seconds,
                        "music_alignment_note": note,
                    }
                )
            )
        else:
            annotated.append(
                scene.model_copy(
                    update={
                        "music_energy": "unknown",
                        "music_section_start_seconds": None,
                        "music_section_end_seconds": None,
                        "music_alignment_note": "No analyzed soundtrack section covers this scene.",
                    }
                )
            )
        scene_start = scene_end
    return annotated
