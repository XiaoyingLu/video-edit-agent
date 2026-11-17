import json

import pytest

from travel_video_agent.artifact_tracking import StaleArtifactError, validate_artifact
from travel_video_agent.editing import planning_pipeline
from travel_video_agent.editing.intent_schemas import EditingPlan


def test_create_video_project_plan_registers_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(
        planning_pipeline,
        "parse_editing_intent",
        lambda prompt, model=None: EditingPlan(
            source_prompt=prompt,
            target_duration_seconds=90,
            style="cinematic",
            priorities=["scenic_landscapes"],
        ),
    )

    plan, destination = planning_pipeline.create_video_project_plan(
        tmp_path, "Create a 90-second cinematic mountain film", model="qwen2.5vl:3b"
    )

    editing_path = tmp_path / ".video-agent" / "editing-plan.json"
    assert destination == tmp_path / ".video-agent" / "video-project-plan.json"
    assert json.loads(destination.read_text(encoding="utf-8")) == plan.model_dump(mode="json")
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
