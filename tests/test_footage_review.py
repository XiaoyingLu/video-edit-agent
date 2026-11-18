"""Tests for human-reviewed storyboard footage assignment."""

import pytest

from travel_video_agent.artifact_tracking import register_artifact
from travel_video_agent.editing.footage_mapping import suggest_storyboard_footage
from travel_video_agent.editing.footage_review import review_storyboard_footage
from travel_video_agent.editing.planning_schemas import (
    CreativeBrief,
    PlannedScene,
    StoryboardShot,
    StoryTreatment,
    VideoProjectPlan,
)
from travel_video_agent.understanding.schemas import (
    FrameObservation,
    VideoScene,
    VideoUnderstanding,
)


def _artifacts(root):
    artifact_dir = root / ".video-agent"
    artifact_dir.mkdir(exist_ok=True)
    video = root / "clip.mp4"
    video.write_bytes(b"test video placeholder")
    key = __import__("hashlib").sha256(b"clip.mp4").hexdigest()[:12]
    plan = VideoProjectPlan(
        project_path=str(root.resolve()),
        brief=CreativeBrief(
            source_prompt="Create a 5-second travel film",
            target_duration_seconds=5,
            style="cinematic",
            aspect_ratio="16:9",
            resolution="1080p",
            pacing="moderate",
        ),
        editing_preferences={
            "source_prompt": "Create a 5-second travel film",
            "target_duration_seconds": 5,
            "style": "cinematic",
        },
        story_treatment=StoryTreatment(
            working_title="Mountain Light",
            logline="A quiet mountain journey.",
            narrative_arc=["Arrival", "Wonder"],
            emotional_journey="Calm becomes awe.",
        ),
        scenes=[
            PlannedScene(
                scene_id="scene-01",
                sequence_index=1,
                title="Mountains",
                purpose="Reveal the landscape",
                narrative_function="hook",
                target_duration_seconds=5,
                shots=[
                    StoryboardShot(
                        shot_id="shot-001",
                        scene_id="scene-01",
                        sequence_index=1,
                        purpose="Reveal mountain scenery",
                        visual_description="A wide view of a mountain lake",
                        target_duration_seconds=5,
                    )
                ],
            )
        ],
        planning_stage="storyboard_ready",
    )
    understanding = VideoUnderstanding(
        project_path=str(root.resolve()),
        video_count=1,
        scene_count=1,
        model="test",
        scenes=[
            VideoScene(
                scene_id=f"{key}-scene-0001",
                start_seconds=0,
                end_seconds=8,
                summary="A mountain lake surrounded by peaks.",
                dynamics={
                    "temporal_summary": "A hiker approaches the lake viewpoint.",
                    "action_progression": ["A hiker approaches the viewpoint"],
                    "subject_motion": "approaching",
                    "camera_motion": "tracking",
                    "confidence": 0.9,
                },
                observations=[
                    FrameObservation(
                        timestamp_seconds=3,
                        frame_path=".video-agent/frames/frame.jpg",
                        description="A wide mountain lake surrounded by peaks.",
                        shot_scale="wide",
                        visual_role="establishing",
                        composition="leading_lines",
                        primary_subject="mountain lake",
                    )
                ],
            )
        ],
    )
    plan_path = artifact_dir / "video-project-plan.json"
    understanding_path = artifact_dir / "understanding.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    understanding_path.write_text(understanding.model_dump_json(), encoding="utf-8")
    register_artifact(root, plan_path)
    register_artifact(root, understanding_path)
    suggestions = suggest_storyboard_footage(root)
    suggestions_path = artifact_dir / "footage-suggestions.json"
    suggestions_path.write_text(suggestions.model_dump_json(), encoding="utf-8")
    register_artifact(root, suggestions_path)
    return plan_path, suggestions_path


def test_review_footage_applies_only_explicit_selection_and_advances_stage(tmp_path):
    plan_path, suggestions_path = _artifacts(tmp_path)
    plan, _ = review_storyboard_footage(
        tmp_path, selections={"shot-001": 1}, plan_path=plan_path, suggestions_path=suggestions_path
    )
    shot = plan.scenes[0].shots[0]
    assert plan.planning_stage == "footage_mapped"
    assert shot.assignment_status == "approved"
    assert shot.footage_path == "clip.mp4"
    assert (shot.footage_start_seconds, shot.footage_end_seconds) == (0, 8)
    assert shot.temporal_summary == "A hiker approaches the lake viewpoint."
    assert shot.action_progression == ["A hiker approaches the viewpoint"]
    assert shot.dynamics_confidence == 0.9
    assert shot.subject_motion == "approaching"
    assert shot.camera_motion == "tracking"
    assert shot.composition == "leading_lines"
    assert shot.primary_subject == "mountain lake"
    assert plan.approved is False


def test_review_footage_does_not_assign_unselected_shots(tmp_path):
    plan_path, suggestions_path = _artifacts(tmp_path)
    plan, _ = review_storyboard_footage(
        tmp_path, selections={}, plan_path=plan_path, suggestions_path=suggestions_path
    )
    assert plan.planning_stage == "storyboard_ready"
    assert plan.scenes[0].shots[0].assignment_status == "unassigned"


def test_review_footage_rejects_unknown_shot(tmp_path):
    plan_path, suggestions_path = _artifacts(tmp_path)
    with pytest.raises(ValueError, match="unknown storyboard shot"):
        review_storyboard_footage(
            tmp_path,
            selections={"missing": 1},
            plan_path=plan_path,
            suggestions_path=suggestions_path,
        )
