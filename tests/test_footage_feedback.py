"""Tests for project-local footage review feedback."""

import pytest

from video_edit_agent.editing.footage_feedback import (
    FootageFeedbackEntry,
    load_footage_feedback,
    record_footage_feedback,
)


def test_record_and_load_feedback_is_project_local(tmp_path):
    entries = [
        FootageFeedbackEntry(
            shot_tokens=["mountain", "lake"],
            shot_scale="wide",
            visual_role="establishing",
            selected_scene_id="scene-1",
            rejected_scene_ids=["scene-2"],
        )
    ]

    path = record_footage_feedback(tmp_path, entries=entries)
    loaded = load_footage_feedback(tmp_path)

    assert path == tmp_path / ".video-agent" / "footage-feedback.json"
    assert loaded.project_path == str(tmp_path.resolve())
    assert loaded.entries == entries


def test_feedback_rejects_path_outside_project(tmp_path):
    with pytest.raises(ValueError, match="inside the project folder"):
        record_footage_feedback(
            tmp_path,
            entries=[FootageFeedbackEntry(shot_tokens=["mountain"])],
            feedback_path=tmp_path.parent / "outside.json",
        )


def test_empty_feedback_does_not_create_artifact(tmp_path):
    assert record_footage_feedback(tmp_path, entries=[]) is None
    assert not (tmp_path / ".video-agent" / "footage-feedback.json").exists()


def test_feedback_schema_rejects_empty_tokens():
    with pytest.raises(ValueError):
        FootageFeedbackEntry(shot_tokens=[])
