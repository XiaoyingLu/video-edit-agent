import json

import pytest

from travel_video_agent.artifact_tracking import (
    StaleArtifactError,
    register_artifact,
    validate_artifact,
)
from travel_video_agent.editing import planning_generation, planning_pipeline
from travel_video_agent.editing.intent_schemas import EditingPlan
from travel_video_agent.editing.planning_generation import (
    GeneratedVideoPlanContent,
    explicit_duration_seconds,
    infer_creative_priorities,
    normalize_storyboard_ids,
)
from travel_video_agent.editing.planning_schemas import (
    PlannedScene,
    StoryboardShot,
    StoryTreatment,
)
from travel_video_agent.music.schemas import MusicAnalysis, MusicSection


def _generated_content():
    return GeneratedVideoPlanContent(
        story_treatment=StoryTreatment(
            working_title="Beyond the Mountains",
            logline="A cinematic journey from first glimpse to quiet reflection in the Rockies.",
            narrative_arc=["Arrival", "Discovery", "Immersion", "Reflection"],
            emotional_journey="Anticipation grows into wonder before settling into reflection.",
        ),
        scenes=[
            PlannedScene(
                scene_id="scene-01",
                sequence_index=1,
                title="First Light",
                purpose="Establish the scale and atmosphere of the destination.",
                narrative_function="hook",
                target_duration_seconds=45,
                shots=[
                    StoryboardShot(
                        shot_id="shot-01",
                        scene_id="scene-01",
                        sequence_index=1,
                        purpose="Reveal the destination",
                        visual_description="A wide mountain panorama at first light.",
                        shot_scale="extreme_wide",
                        visual_role="establishing",
                        target_duration_seconds=45,
                    )
                ],
            ),
            PlannedScene(
                scene_id="scene-02",
                sequence_index=2,
                title="Into the Landscape",
                purpose="Move from grand scenery into intimate travel details.",
                narrative_function="development",
                target_duration_seconds=45,
                shots=[
                    StoryboardShot(
                        shot_id="shot-02",
                        scene_id="scene-02",
                        sequence_index=1,
                        purpose="Show the journey",
                        visual_description="A traveler approaches a turquoise alpine lake.",
                        shot_scale="wide",
                        visual_role="action",
                        target_duration_seconds=45,
                    )
                ],
            ),
        ],
    )


def test_create_video_project_plan_generates_storyboard_and_registers_dependency(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        planning_pipeline,
        "parse_editing_intent",
        lambda prompt, model=None, memory_override_env=None: EditingPlan(
            source_prompt=prompt,
            target_duration_seconds=60,
            style="cinematic",
            priorities=["scenic_landscapes"],
        ),
    )
    monkeypatch.setattr(
        planning_pipeline,
        "generate_video_plan_content",
        lambda prompt, brief, model=None, music_sections=None: _generated_content(),
    )

    plan, destination = planning_pipeline.create_video_project_plan(
        tmp_path, "Create a 90-second cinematic mountain film", model="qwen2.5vl:3b"
    )

    editing_path = tmp_path / ".video-agent" / "editing-plan.json"
    assert destination == tmp_path / ".video-agent" / "video-project-plan.json"
    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert saved == plan.model_dump(mode="json")
    assert plan.brief.target_duration_seconds == 90
    assert plan.story_treatment.working_title == "Beyond the Mountains"
    assert len(plan.scenes) == 2
    assert plan.planning_stage == "storyboard_ready"
    assert sum(s.target_duration_seconds for scene in plan.scenes for s in scene.shots) == 90
    assert validate_artifact(tmp_path, destination) is True

    editing_path.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(StaleArtifactError, match="modified after it was generated"):
        validate_artifact(tmp_path, destination)


def test_create_video_project_plan_rejects_output_overwriting_dependency(tmp_path):
    with pytest.raises(ValueError, match="cannot overwrite editing-plan.json"):
        planning_pipeline.create_video_project_plan(
            tmp_path,
            "Create a travel video",
            output_path=tmp_path / ".video-agent" / "editing-plan.json",
        )


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Create a 90-second travel film", 90),
        ("Create a video that is 2 minutes long", 120),
        ("Make a 1.5 hour documentary", 5400),
        ("Create a cinematic travel film", None),
    ],
)
def test_explicit_duration_is_extracted_from_prompt(prompt, expected):
    assert explicit_duration_seconds(prompt) == expected


def test_prompt_keywords_add_scenic_priority():
    priorities = infer_creative_priorities(
        "Create a cinematic Banff film about mountain scenery and lakes",
        ["memorable_moments"],
    )
    assert priorities == ["memorable_moments", "scenic_landscapes"]


def test_normalize_storyboard_ids_repairs_duplicate_model_ids():
    content = _generated_content()
    duplicate_scene = content.scenes[1].model_copy(
        update={
            "scene_id": content.scenes[0].scene_id,
            "sequence_index": 1,
            "shots": [
                content.scenes[1]
                .shots[0]
                .model_copy(update={"shot_id": content.scenes[0].shots[0].shot_id})
            ],
        }
    )
    normalized = normalize_storyboard_ids(
        content.model_copy(update={"scenes": [content.scenes[0], duplicate_scene]})
    )

    assert [scene.scene_id for scene in normalized.scenes] == ["scene-01", "scene-02"]
    assert [scene.sequence_index for scene in normalized.scenes] == [1, 2]
    shots = [shot for scene in normalized.scenes for shot in scene.shots]
    assert [shot.shot_id for shot in shots] == ["shot-001", "shot-002"]
    assert [shot.scene_id for shot in shots] == ["scene-01", "scene-02"]


def test_planner_discards_partial_model_footage_assignments(monkeypatch):
    generated = _generated_content().model_dump(mode="json")
    generated["scenes"][0]["music_energy"] = "high"
    generated["scenes"][0]["music_section_start_seconds"] = 0
    generated["scenes"][0]["music_section_end_seconds"] = 45
    first_shot = generated["scenes"][0]["shots"][0]
    first_shot["footage_start_seconds"] = 12.5
    first_shot["footage_path"] = "invented.mp4"
    first_shot["assignment_status"] = "suggested"

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"message": {"content": json.dumps(generated)}}).encode()

    monkeypatch.setattr(planning_generation, "ensure_memory_available", lambda *args: None)
    monkeypatch.setattr(planning_generation, "urlopen", lambda *args, **kwargs: FakeResponse())
    planner = planning_generation.OllamaVideoPlanner(model="qwen2.5:1.5b")

    result = planner.generate("Create a cinematic travel film", {"target_duration_seconds": 90})

    shot = result.scenes[0].shots[0]
    assert shot.footage_path is None
    assert shot.footage_start_seconds is None
    assert shot.footage_end_seconds is None
    assert shot.assignment_status == "unassigned"
    assert result.scenes[0].music_energy == "unknown"
    assert result.scenes[0].music_section_start_seconds is None


def test_create_video_project_plan_uses_music_sections_and_tracks_dependency(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        planning_pipeline,
        "parse_editing_intent",
        lambda prompt, model=None, memory_override_env=None: EditingPlan(
            source_prompt=prompt,
            target_duration_seconds=90,
            style="cinematic",
            priorities=["scenic_landscapes"],
        ),
    )
    captured = {}

    def fake_generate(prompt, brief, model=None, music_sections=None):
        captured["sections"] = music_sections
        return _generated_content()

    monkeypatch.setattr(planning_pipeline, "generate_video_plan_content", fake_generate)
    artifact_dir = tmp_path / ".video-agent"
    artifact_dir.mkdir()
    soundtrack = tmp_path / "travel-score.wav"
    soundtrack.write_bytes(b"audio")
    analysis = MusicAnalysis(
        project_path=str(tmp_path.resolve()),
        music_path=str(soundtrack.resolve()),
        duration_seconds=90,
        sample_rate=11025,
        bpm=120,
        beat_confidence=0.8,
        beat_interval_seconds=0.5,
        beat_times_seconds=[0.5, 1.0],
        downbeat_times_seconds=[0.5],
        sections=[
            MusicSection(start_seconds=0, end_seconds=30, energy="low", energy_rms=0.1),
            MusicSection(start_seconds=30, end_seconds=60, energy="high", energy_rms=0.4),
            MusicSection(start_seconds=60, end_seconds=90, energy="medium", energy_rms=0.2),
        ],
    )
    analysis_path = artifact_dir / "music-analysis.json"
    analysis_path.write_text(analysis.model_dump_json(), encoding="utf-8")
    register_artifact(tmp_path, analysis_path, dependencies=[soundtrack])

    plan, destination = planning_pipeline.create_video_project_plan(
        tmp_path,
        "Create a 90-second cinematic mountain film",
        music_analysis_path=analysis_path,
    )

    assert [item["energy"] for item in captured["sections"]] == ["low", "high", "medium"]
    assert plan.source_music_analysis == ".video-agent/music-analysis.json"
    assert [scene.music_energy for scene in plan.scenes] == ["low", "medium"]
    assert plan.scenes[0].music_section_start_seconds == 0
    assert plan.scenes[1].music_section_start_seconds == 60
    assert validate_artifact(tmp_path, destination) is True

    analysis_path.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(StaleArtifactError):
        validate_artifact(tmp_path, destination)


def test_create_video_project_plan_rejects_external_music_analysis(tmp_path):
    external = tmp_path.parent / "music-analysis.json"
    external.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inside the project folder"):
        planning_pipeline.create_video_project_plan(
            tmp_path,
            "Create a cinematic travel film",
            music_analysis_path=external,
        )


def test_planned_scene_validates_music_section_bounds():
    scene = _generated_content().scenes[0]
    with pytest.raises(ValueError, match="must be supplied together"):
        PlannedScene.model_validate(
            scene.model_dump() | {"music_section_start_seconds": 2.0}
        )


def test_scene_music_annotation_uses_largest_overlap_and_marks_uncovered():
    scenes = _generated_content().scenes
    annotated = planning_pipeline._annotate_scenes_with_music(
        scenes,
        [
            MusicSection(start_seconds=0, end_seconds=20, energy="low", energy_rms=0.1),
            MusicSection(start_seconds=20, end_seconds=45, energy="high", energy_rms=0.4),
        ],
    )

    assert annotated[0].music_energy == "high"
    assert annotated[0].music_section_start_seconds == 20
    assert annotated[1].music_energy == "unknown"
    assert annotated[1].music_section_start_seconds is None
