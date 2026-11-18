from pathlib import Path

import pytest

from travel_video_agent.editing.intent_schemas import EditingPlan
from travel_video_agent.editing.planning_schemas import VideoProjectPlan
from travel_video_agent.editing.storyboard_review import generate_storyboard_review
from travel_video_agent.understanding.schemas import VideoUnderstanding


def _artifacts(root: Path) -> None:
    artifact_dir = root / ".video-agent"
    artifact_dir.mkdir()
    plan = VideoProjectPlan(
        project_path=str(root.resolve()),
        brief={
            "source_prompt": "Create a 60-second cinematic travel film",
            "target_duration_seconds": 60,
            "style": "cinematic",
            "aspect_ratio": "16:9",
            "resolution": "1080p",
            "pacing": "moderate",
        },
        editing_preferences=EditingPlan(source_prompt="Create a 60-second cinematic travel film"),
        story_treatment={
            "working_title": "A <Beautiful> Journey",
            "logline": "A journey from arrival to discovery.",
            "narrative_arc": ["Arrival", "Discovery"],
            "emotional_journey": "Curiosity becomes wonder.",
        },
        scenes=[
            {
                "scene_id": "scene-01",
                "sequence_index": 1,
                "title": "The Arrival",
                "purpose": "Introduce the destination.",
                "narrative_function": "hook",
                "target_duration_seconds": 60,
                "shots": [
                    {
                        "shot_id": "shot-001",
                        "scene_id": "scene-01",
                        "sequence_index": 1,
                        "purpose": "Reveal the landscape",
                        "visual_description": "A wide mountain panorama",
                        "target_duration_seconds": 60,
                    }
                ],
            }
        ],
        planning_stage="storyboard_ready",
    )
    understanding = VideoUnderstanding(
        project_path=str(root.resolve()),
        video_count=0,
        scene_count=0,
        scenes=[],
        model="qwen2.5vl:3b",
    )
    (artifact_dir / "video-project-plan.json").write_text(plan.model_dump_json(), encoding="utf-8")
    (artifact_dir / "understanding.json").write_text(
        understanding.model_dump_json(), encoding="utf-8"
    )


def test_generate_storyboard_review_writes_read_only_html(tmp_path: Path) -> None:
    _artifacts(tmp_path)

    destination, shot_count, image_count = generate_storyboard_review(tmp_path)

    document = destination.read_text(encoding="utf-8")
    assert destination == tmp_path / ".video-agent" / "storyboard.html"
    assert shot_count == 1
    assert image_count == 0
    assert "shot-001" in document
    assert "A &lt;Beautiful&gt; Journey" in document
    assert "No representative frame available" in document
    assert "does not approve the project" in document


def test_storyboard_review_rejects_output_outside_project(tmp_path: Path) -> None:
    _artifacts(tmp_path)

    with pytest.raises(ValueError, match="inside the project folder"):
        generate_storyboard_review(tmp_path, output_path=tmp_path.parent / "storyboard.html")
