import hashlib
import json
from pathlib import Path

from typer.testing import CliRunner

from travel_video_agent.cli import app, understanding
from travel_video_agent.editing import planning_pipeline
from travel_video_agent.editing.intent_schemas import EditingPlan
from travel_video_agent.editing.planning_generation import GeneratedVideoPlanContent
from travel_video_agent.editing.planning_schemas import (
    PlannedScene,
    StoryTreatment,
    StoryboardShot,
)
from travel_video_agent.understanding.schemas import (
    FrameObservation,
    VideoScene,
    VideoUnderstanding,
)

runner = CliRunner()


def test_understand_create_plan_suggest_footage_smoke(tmp_path: Path, monkeypatch) -> None:
    """Exercise the CLI artifact hand-offs with deterministic, stubbed model outputs."""
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"smoke-test-video")
    video_key = hashlib.sha256(b"clip.mp4").hexdigest()[:12]
    scene_id = f"{video_key}-scene-0001"
    frame = tmp_path / "representative.jpg"
    frame.write_bytes(b"stub-frame")

    understanding_result = VideoUnderstanding(
        project_path=str(tmp_path.resolve()),
        video_count=1,
        scene_count=1,
        model="stub-vision-model",
        scenes=[
            VideoScene(
                scene_id=scene_id,
                start_seconds=0,
                end_seconds=10,
                summary="A hiker walks beside an alpine lake.",
                story_function="action",
                observations=[
                    FrameObservation(
                        timestamp_seconds=5,
                        frame_path=str(frame),
                        description="A hiker walking beside a turquoise alpine lake.",
                        objects=["hiker", "lake", "mountains"],
                        activities=["walking"],
                        primary_subject="hiker",
                        setting="mountain lake",
                        visual_role="action",
                    )
                ],
            )
        ],
    )
    monkeypatch.setattr(
        understanding, "understand_folder", lambda *args, **kwargs: understanding_result
    )

    def parse_intent(prompt, model=None, memory_override_env=None):
        return EditingPlan(
            source_prompt=prompt,
            target_duration_seconds=10,
            style="cinematic",
            priorities=["scenic_landscapes"],
        )

    def generate_plan(prompt, brief, model=None, music_sections=None, story_chapters=None):
        return GeneratedVideoPlanContent(
            story_treatment=StoryTreatment(
                working_title="Along the Alpine Lake",
                logline="A quiet walk reveals a mountain lake.",
                narrative_arc=["Arrival", "Discovery"],
                emotional_journey="Curiosity settles into calm.",
            ),
            scenes=[
                PlannedScene(
                    scene_id="scene-01",
                    sequence_index=1,
                    title="By the Water",
                    purpose="Establish the landscape and the traveler.",
                    narrative_function="establishing",
                    target_duration_seconds=10,
                    shots=[
                        StoryboardShot(
                            shot_id="shot-01",
                            scene_id="scene-01",
                            sequence_index=1,
                            purpose="Show a traveler beside the lake",
                            visual_description="A hiker walking beside a turquoise alpine lake.",
                            shot_scale="wide",
                            visual_role="establishing",
                            target_duration_seconds=10,
                        )
                    ],
                )
            ],
        )

    monkeypatch.setattr(planning_pipeline, "parse_editing_intent", parse_intent)
    monkeypatch.setattr(planning_pipeline, "generate_video_plan_content", generate_plan)

    result = runner.invoke(app, ["understand", str(tmp_path)])
    assert result.exit_code == 0, result.output
    understanding_path = tmp_path / ".video-agent" / "understanding.json"
    assert json.loads(understanding_path.read_text())["scene_count"] == 1

    result = runner.invoke(
        app,
        ["create-plan", str(tmp_path), "--prompt", "Create a 10-second cinematic video"],
    )
    assert result.exit_code == 0, result.output
    plan_path = tmp_path / ".video-agent" / "video-project-plan.json"
    assert json.loads(plan_path.read_text())["planning_stage"] == "storyboard_ready"

    result = runner.invoke(app, ["suggest-footage", str(tmp_path)])
    assert result.exit_code == 0, result.output
    suggestions_path = tmp_path / ".video-agent" / "footage-suggestions.json"
    suggestions = json.loads(suggestions_path.read_text())
    assert suggestions["matched_shot_count"] == 1
    assert suggestions["candidate_count"] >= 1
    assert suggestions["suggestions"][0]["candidates"][0]["source_scene_id"] == scene_id
