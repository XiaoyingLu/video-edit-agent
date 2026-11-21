import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from video_edit_agent.cli import app
from video_edit_agent.editing import planning_pipeline
from video_edit_agent.editing.intent_schemas import EditingPlan
from video_edit_agent.editing.planning_generation import GeneratedVideoPlanContent
from video_edit_agent.editing.planning_schemas import (
    PlannedScene,
    StoryboardShot,
    StoryTreatment,
)
from video_edit_agent.understanding import pipeline as understanding_pipeline

runner = CliRunner()


def test_understand_create_plan_suggest_footage_smoke(tmp_path: Path, monkeypatch) -> None:
    """Run a real tiny clip through the CLI stages with deterministic model responses."""
    source = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x120:r=12:d=2",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(source),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )

    class FakeVisionAnalyzer:
        def __init__(self, model: str) -> None:
            self.model = model

        def analyze_frame(self, frame_path: Path) -> dict:
            return {
                "description": "A hiker walks beside a turquoise alpine lake.",
                "objects": ["hiker", "lake", "mountains"],
                "activities": ["walking"],
                "confidence": 0.95,
                "shot_scale": "wide",
                "visual_role": "establishing",
                "setting": "mountain lake",
                "primary_subject": "hiker",
            }

        def unload_model(self) -> bool:
            return True

    monkeypatch.setattr(understanding_pipeline, "OllamaVisionAnalyzer", FakeVisionAnalyzer)

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
                    target_duration_seconds=5,
                    shots=[
                        StoryboardShot(
                            shot_id="shot-01",
                            scene_id="scene-01",
                            sequence_index=1,
                            purpose="Show a traveler beside the lake",
                            visual_description="A hiker walking beside a turquoise alpine lake.",
                            shot_scale="wide",
                            visual_role="establishing",
                            target_duration_seconds=5,
                        )
                    ],
                ),
                PlannedScene(
                    scene_id="scene-02",
                    sequence_index=2,
                    title="A Quiet Pause",
                    purpose="Let the landscape breathe.",
                    narrative_function="scenic_breath",
                    target_duration_seconds=5,
                    shots=[
                        StoryboardShot(
                            shot_id="shot-02",
                            scene_id="scene-02",
                            sequence_index=1,
                            purpose="Hold on the alpine lake",
                            visual_description="A calm wide view of a turquoise alpine lake.",
                            shot_scale="wide",
                            visual_role="scenic_breath",
                            target_duration_seconds=5,
                        )
                    ],
                ),
            ],
        )

    monkeypatch.setattr(planning_pipeline, "parse_editing_intent", parse_intent)
    monkeypatch.setattr(planning_pipeline, "generate_video_plan_content", generate_plan)

    result = runner.invoke(
        app, ["understand", str(tmp_path), "--provider", "ollama", "--samples-per-scene", "1"]
    )
    assert result.exit_code == 0, result.output
    understanding_path = tmp_path / ".video-agent" / "understanding.json"
    assert json.loads(understanding_path.read_text())["scene_count"] >= 1

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
    assert suggestions["matched_shot_count"] >= 1
    assert suggestions["candidate_count"] >= 1
    assert suggestions["suggestions"][0]["candidates"][0]["video_path"] == "clip.mp4"
