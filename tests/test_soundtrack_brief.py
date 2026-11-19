import json
from pathlib import Path

import pytest

from travel_video_agent.editing.planning_schemas import VideoProjectPlan
from travel_video_agent.soundtrack.brief import (
    build_soundtrack_brief,
    create_soundtrack_brief,
)


def _plan(root: Path) -> VideoProjectPlan:
    return VideoProjectPlan(
        project_path=str(root.resolve()),
        brief={
            "source_prompt": "Create a 30-second cinematic mountain journey full of wonder",
            "target_duration_seconds": 30,
            "style": "cinematic",
            "aspect_ratio": "16:9",
            "resolution": "1080p",
            "pacing": "moderate",
            "creative_priorities": ["scenic_landscapes"],
        },
        editing_preferences={
            "source_prompt": "Create a 30-second cinematic mountain journey full of wonder",
            "target_duration_seconds": 30,
            "style": "cinematic",
            "priorities": ["scenic_landscapes"],
        },
        story_treatment={
            "working_title": "Mountain Light",
            "logline": "A journey through quiet alpine landscapes.",
            "narrative_arc": ["Arrival", "Discovery", "Reflection"],
            "emotional_journey": "Anticipation grows into wonder, then calm.",
        },
        scenes=[
            {
                "scene_id": "scene-1",
                "sequence_index": 1,
                "title": "Arrival",
                "purpose": "Establish the mountain landscape.",
                "narrative_function": "establishing",
                "target_duration_seconds": 10,
                "emotional_intent": "Anticipation and wonder",
                "shots": [
                    {
                        "shot_id": "shot-1",
                        "scene_id": "scene-1",
                        "sequence_index": 1,
                        "purpose": "Reveal the mountains",
                        "visual_description": "Wide mountain panorama",
                        "target_duration_seconds": 10,
                    }
                ],
            },
            {
                "scene_id": "scene-2",
                "sequence_index": 2,
                "title": "Summit",
                "purpose": "Reach the emotional peak.",
                "narrative_function": "climax",
                "target_duration_seconds": 10,
                "emotional_intent": "Exciting discovery",
                "shots": [
                    {
                        "shot_id": "shot-2",
                        "scene_id": "scene-2",
                        "sequence_index": 1,
                        "purpose": "Show the summit",
                        "visual_description": "Aerial view of the summit",
                        "target_duration_seconds": 10,
                    }
                ],
            },
            {
                "scene_id": "scene-3",
                "sequence_index": 3,
                "title": "Reflection",
                "purpose": "End with a quiet moment.",
                "narrative_function": "resolution",
                "target_duration_seconds": 10,
                "emotional_intent": "Peaceful reflection",
                "shots": [
                    {
                        "shot_id": "shot-3",
                        "scene_id": "scene-3",
                        "sequence_index": 1,
                        "purpose": "Hold on the lake",
                        "visual_description": "Still alpine lake",
                        "target_duration_seconds": 10,
                    }
                ],
            },
        ],
        planning_stage="storyboard_ready",
    )


def test_build_soundtrack_brief_uses_story_arc_and_rights_defaults(tmp_path: Path) -> None:
    brief = build_soundtrack_brief(_plan(tmp_path), ".video-agent/video-project-plan.json")

    assert brief.target_duration_seconds == 30
    assert brief.mood
    assert brief.energy_arc[0].energy == "medium"
    assert brief.energy_arc[1].energy == "high"
    assert brief.energy_arc[2].energy == "low"
    assert brief.tempo_min_bpm < brief.tempo_max_bpm
    assert brief.rights_policy.commercial_use_required is True
    assert brief.rights_policy.monetized_video_allowed_required is True
    assert "instrumental" in brief.search_terms
    assert "Avoid vocals" in brief.generation_prompt


def test_create_soundtrack_brief_writes_tracked_artifact(tmp_path: Path) -> None:
    artifact_dir = tmp_path / ".video-agent"
    artifact_dir.mkdir()
    source = artifact_dir / "video-project-plan.json"
    source.write_text(_plan(tmp_path).model_dump_json(), encoding="utf-8")

    brief, destination = create_soundtrack_brief(tmp_path)

    assert destination == artifact_dir / "soundtrack-brief.json"
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved == brief.model_dump(mode="json")
    assert saved["source_video_plan"] == ".video-agent/video-project-plan.json"
    assert saved["rights_policy"]["commercial_use_required"] is True


def test_create_soundtrack_brief_rejects_plan_from_another_project(tmp_path: Path) -> None:
    artifact_dir = tmp_path / ".video-agent"
    artifact_dir.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    source = artifact_dir / "video-project-plan.json"
    source.write_text(_plan(other).model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="different project"):
        create_soundtrack_brief(tmp_path)


def test_create_soundtrack_brief_rejects_output_outside_project(tmp_path: Path) -> None:
    artifact_dir = tmp_path / ".video-agent"
    artifact_dir.mkdir()
    source = artifact_dir / "video-project-plan.json"
    source.write_text(_plan(tmp_path).model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="inside the project"):
        create_soundtrack_brief(tmp_path, output_path=tmp_path.parent / "outside.json")
