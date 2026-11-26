from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException

from video_edit_agent import web_api


def _create_project(tmp_path: Path, monkeypatch) -> dict:
    monkeypatch.setattr(web_api, "PROJECTS_DIR", tmp_path)
    project = web_api.create_project(web_api.ProjectCreate(name="Test project"))
    project["media"] = [
        {
            "name": "clip.mp4",
            "path": str(tmp_path / project["id"] / "source/clip.mp4"),
        }
    ]
    web_api._write_project(tmp_path / project["id"], project)
    return project


def test_start_project_analysis_queues_background_task(tmp_path: Path, monkeypatch) -> None:
    project = _create_project(tmp_path, monkeypatch)
    tasks = BackgroundTasks()

    result = web_api.start_project_analysis(project["id"], tasks)

    assert result["analysis_status"] == "queued"
    assert len(tasks.tasks) == 1
    saved = web_api.get_project(project["id"])
    assert saved["analysis_status"] == "queued"


def test_start_project_analysis_requires_uploaded_media(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(web_api, "PROJECTS_DIR", tmp_path)
    project = web_api.create_project(web_api.ProjectCreate(name="Empty project"))

    with pytest.raises(HTTPException) as exc_info:
        web_api.start_project_analysis(project["id"], BackgroundTasks())
    assert exc_info.value.status_code == 400


def test_background_analysis_persists_result_and_status(tmp_path: Path, monkeypatch) -> None:
    project = _create_project(tmp_path, monkeypatch)

    def fake_understand_folder(folder, *, provider, progress_callback):
        assert folder == tmp_path / project["id"]
        assert provider == "ollama"
        progress_callback("scene_detection_completed", {"scene_count": 1})
        return SimpleNamespace(
            video_count=1,
            scene_count=1,
            model_dump=lambda mode: {"video_count": 1, "scene_count": 1, "scenes": []},
        )

    monkeypatch.setattr(web_api, "get_setting", lambda name: "ollama")
    monkeypatch.setattr(web_api, "understand_folder", fake_understand_folder)

    web_api._run_project_analysis(project["id"])

    saved = web_api.get_project(project["id"])
    assert saved["analysis_status"] == "completed"
    assert saved["analysis_progress"]["scene_count"] == 1
    assert web_api.get_project_analysis(project["id"])["scene_count"] == 1


def test_save_edit_decisions_validates_and_persists(tmp_path: Path, monkeypatch) -> None:
    project = _create_project(tmp_path, monkeypatch)
    project_path = tmp_path / project["id"]
    source = project_path / "source" / "clip.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"video")
    artifact = project_path / ".video-agent" / "understanding.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        '{"scenes":[{"scene_id":"scene-1","source_video_path":"source/clip.mp4",'
        '"start_seconds":2.0,"end_seconds":8.0}]}',
        encoding="utf-8",
    )

    request = web_api.EditDecisions(
        clips=[
            web_api.EditDecision(
                clip_id="clip-1",
                scene_id="scene-1",
                source_video_path="source/clip.mp4",
                start_seconds=3.0,
                end_seconds=7.0,
                included=True,
            )
        ]
    )
    result = web_api.save_edit_decisions(project["id"], request)

    assert result["clips"][0]["start_seconds"] == 3.0
    saved = web_api.get_edit_decisions(project["id"])
    assert saved == result


def test_save_edit_decisions_rejects_interval_outside_detected_scene(
    tmp_path: Path, monkeypatch
) -> None:
    project = _create_project(tmp_path, monkeypatch)
    project_path = tmp_path / project["id"]
    source = project_path / "source" / "clip.mp4"
    source.write_bytes(b"video")
    artifact = project_path / ".video-agent" / "understanding.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        '{"scenes":[{"scene_id":"scene-1","source_video_path":"source/clip.mp4",'
        '"start_seconds":2.0,"end_seconds":8.0}]}',
        encoding="utf-8",
    )
    request = web_api.EditDecisions(
        clips=[
            web_api.EditDecision(
                clip_id="clip-1",
                scene_id="scene-1",
                source_video_path="source/clip.mp4",
                start_seconds=1.0,
                end_seconds=7.0,
            )
        ]
    )

    with pytest.raises(HTTPException) as exc_info:
        web_api.save_edit_decisions(project["id"], request)
    assert exc_info.value.status_code == 400
