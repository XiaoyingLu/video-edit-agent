import hashlib
import json
from pathlib import Path

import pytest

from video_edit_agent.artifact_tracking import register_artifact
from video_edit_agent.editing.footage_mapping import suggest_storyboard_footage
from video_edit_agent.editing.footage_visual_review import (
    generate_footage_visual_review,
    load_footage_review_selections,
)
from video_edit_agent.editing.intent_schemas import EditingPlan
from video_edit_agent.editing.planning_schemas import VideoProjectPlan
from video_edit_agent.understanding.schemas import FrameObservation, VideoScene, VideoUnderstanding


def _artifacts(root: Path) -> tuple[Path, Path]:
    artifact_dir = root / ".video-agent"
    artifact_dir.mkdir()
    video = root / "clip.mp4"
    video.write_bytes(b"placeholder")
    key = hashlib.sha256(b"clip.mp4").hexdigest()[:12]
    plan = VideoProjectPlan(
        project_path=str(root.resolve()),
        brief={
            "source_prompt": "Create a short travel film",
            "target_duration_seconds": 5,
            "style": "cinematic",
            "aspect_ratio": "16:9",
            "resolution": "1080p",
            "pacing": "moderate",
        },
        editing_preferences=EditingPlan(source_prompt="Create a short travel film"),
        story_treatment={
            "working_title": "Mountain Light",
            "logline": "A quiet mountain journey.",
            "narrative_arc": ["Arrival", "Wonder"],
            "emotional_journey": "Calm becomes awe.",
        },
        scenes=[
            {
                "scene_id": "scene-01",
                "sequence_index": 1,
                "title": "Mountains",
                "purpose": "Reveal the landscape.",
                "narrative_function": "hook",
                "target_duration_seconds": 5,
                "shots": [
                    {
                        "shot_id": "shot-001",
                        "scene_id": "scene-01",
                        "sequence_index": 1,
                        "purpose": "Reveal mountain scenery",
                        "visual_description": "A wide view of a mountain lake",
                        "target_duration_seconds": 5,
                    }
                ],
            }
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
                observations=[
                    FrameObservation(
                        timestamp_seconds=3,
                        frame_path=".video-agent/frames/frame.jpg",
                        description="A wide mountain lake surrounded by peaks.",
                        shot_scale="wide",
                        visual_role="establishing",
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


def test_generate_footage_visual_review_writes_candidate_choices(tmp_path: Path) -> None:
    plan_path, suggestions_path = _artifacts(tmp_path)

    destination, shot_count, candidate_count = generate_footage_visual_review(
        tmp_path, plan_path=plan_path, suggestions_path=suggestions_path
    )

    document = destination.read_text(encoding="utf-8")
    assert destination == tmp_path / ".video-agent" / "footage-review.html"
    assert shot_count == 1
    assert candidate_count == 1
    assert "Option 1" in document
    assert "Leave this shot unassigned" in document
    assert "Download selections" in document
    assert "nothing is approved or rendered" in document.lower()


def test_load_footage_review_selections_requires_every_shot(tmp_path: Path) -> None:
    path = tmp_path / "selections.json"
    suggestions_path = tmp_path / "footage-suggestions.json"
    suggestions_path.write_text("current suggestions", encoding="utf-8")
    digest = hashlib.sha256(suggestions_path.read_bytes()).hexdigest()
    path.write_text(
        json.dumps({
            "schema_version": "1.0",
            "suggestions_sha256": digest,
            "selections": {"shot-001": 2, "shot-002": 0},
        }),
        encoding="utf-8",
    )

    selected, rejected = load_footage_review_selections(
        path,
        expected_shot_ids={"shot-001", "shot-002"},
        suggestions_path=suggestions_path,
    )

    assert selected == {"shot-001": 2}
    assert rejected == {"shot-002"}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"schema_version": "2.0", "selections": {"shot-001": 1}},
        {"schema_version": "1.0", "selections": {"shot-001": "1"}},
        {"schema_version": "1.0", "selections": {"shot-001": 11}},
    ],
)
def test_load_footage_review_selections_rejects_invalid_exports(tmp_path: Path, payload) -> None:
    path = tmp_path / "selections.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_footage_review_selections(
            path, expected_shot_ids={"shot-001"}, suggestions_path=tmp_path / "suggestions.json"
        )
