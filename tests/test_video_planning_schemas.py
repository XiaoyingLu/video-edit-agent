import json

import pytest
from pydantic import ValidationError

from travel_video_agent.editing.planning_schemas import (
    CreativeBrief,
    StoryboardShot,
    VideoProjectPlan,
)
from travel_video_agent.editing.intent_schemas import EditingPlan


def _brief():
    return CreativeBrief(
        source_prompt="Create a short cinematic trip film",
        target_duration_seconds=60,
        style="cinematic",
        aspect_ratio="16:9",
        resolution="1080p",
        pacing="moderate",
        creative_priorities=["scenic_landscapes"],
    )


def test_video_project_plan_starts_at_brief_stage_without_invented_story():
    editing = EditingPlan(source_prompt="Create a short cinematic trip film")
    plan = VideoProjectPlan(
        project_path="/tmp/trip",
        brief=_brief(),
        editing_preferences=editing,
    )

    assert plan.planning_stage == "brief_ready"
    assert plan.story_treatment is None
    assert plan.scenes == []
    assert plan.approval_required is True
    assert plan.approved is False
    assert plan.rendering_performed is False


def test_video_project_plan_rejects_non_contiguous_scene_order():
    editing = EditingPlan(source_prompt="Create a short cinematic trip film")
    with pytest.raises(ValidationError, match="contiguous"):
        VideoProjectPlan(
            project_path="/tmp/trip",
            brief=_brief(),
            editing_preferences=editing,
            planning_stage="story_ready",
            story_treatment={
                "working_title": "A journey",
                "logline": "A journey through the mountains.",
                "narrative_arc": ["departure", "discovery"],
                "emotional_journey": "Curiosity to wonder",
            },
            scenes=[
                {
                    "scene_id": "scene-1",
                    "sequence_index": 2,
                    "title": "Opening",
                    "purpose": "Introduce the place",
                    "narrative_function": "hook",
                    "target_duration_seconds": 10,
                }
            ],
        )


def test_storyboard_shot_requires_complete_footage_interval():
    with pytest.raises(ValidationError, match="supplied together"):
        StoryboardShot(
            shot_id="shot-1",
            scene_id="scene-1",
            sequence_index=1,
            purpose="Establish the destination",
            visual_description="A wide mountain view",
            target_duration_seconds=5,
            footage_path="trip.mp4",
            footage_start_seconds=1,
            assignment_status="suggested",
        )


def test_video_project_plan_round_trips_json():
    editing = EditingPlan(source_prompt="Create a short cinematic trip film")
    plan = VideoProjectPlan(
        project_path="/tmp/trip",
        brief=_brief(),
        editing_preferences=editing,
    )
    restored = VideoProjectPlan.model_validate_json(plan.model_dump_json())
    assert restored == plan
