import json

import pytest

from travel_video_agent.artifact_tracking import StaleArtifactError, validate_artifact
from travel_video_agent.editing import planning_pipeline
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
        lambda prompt, brief, model=None: _generated_content(),
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
                content.scenes[1].shots[0].model_copy(
                    update={"shot_id": content.scenes[0].shots[0].shot_id}
                )
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
