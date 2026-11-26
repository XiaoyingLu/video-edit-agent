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
    monkeypatch.setattr(
        web_api,
        "probe_runtime_capabilities",
        lambda _path: SimpleNamespace(require=lambda *_names: None),
    )
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


def test_edit_decision_put_allows_frontend_cors_preflight() -> None:
    from fastapi.testclient import TestClient

    response = TestClient(web_api.app).options(
        "/api/projects/example/edit-decisions",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "PUT",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert "PUT" in response.headers["access-control-allow-methods"]


def test_export_project_renders_included_clips_and_returns_mp4(
    tmp_path, monkeypatch
) -> None:
    from fastapi.testclient import TestClient

    project = _create_project(tmp_path, monkeypatch)
    project_path = tmp_path / project["id"]
    source = project_path / "source" / "clip.mp4"
    source.write_bytes(b"video")
    artifact = project_path / ".video-agent" / "understanding.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        '{"scenes":[{"scene_id":"scene-1","source_video_path":"source/clip.mp4",'
        '"start_seconds":0.0,"end_seconds":10.0,"summary":"Mountain lake"}]}',
        encoding="utf-8",
    )

    def fake_render(
        folder, *, timeline_path, output_path, output_aspect_ratio, output_resolution
    ):
        from video_edit_agent.editing.timeline_schemas import ProjectDraftTimeline

        timeline = ProjectDraftTimeline.model_validate_json(timeline_path.read_text())
        assert timeline.timeline_approved is True
        assert [clip.clip_id for clip in timeline.clips] == ["clip-1"]
        assert timeline.aspect_ratio == "9:16"
        assert output_aspect_ratio == "9:16"
        assert output_resolution == "720p"
        output_path.write_bytes(b"rendered-mp4")

    monkeypatch.setattr(web_api, "render_approved_shot_timeline", fake_render)
    response = TestClient(web_api.app).post(
        f"/api/projects/{project['id']}/export",
        json={
            "aspect_ratio": "9:16",
            "resolution": "720p",
            "clips": [
                {
                    "clip_id": "clip-1",
                    "scene_id": "scene-1",
                    "source_video_path": "source/clip.mp4",
                    "start_seconds": 2.0,
                    "end_seconds": 8.0,
                    "included": True,
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("video/mp4")
    assert response.content == b"rendered-mp4"


def test_export_project_requires_at_least_one_included_clip(tmp_path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    project = _create_project(tmp_path, monkeypatch)
    response = TestClient(web_api.app).post(
        f"/api/projects/{project['id']}/export",
        json={
            "clips": [
                {
                    "clip_id": "clip-1",
                    "scene_id": "scene-1",
                    "source_video_path": "source/clip.mp4",
                    "start_seconds": 1.0,
                    "end_seconds": 2.0,
                    "included": False,
                }
            ],
        },
    )

    assert response.status_code == 400
    assert "at least one scene" in response.json()["detail"]


def test_get_project_media_serves_uploaded_video(tmp_path: Path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    project = _create_project(tmp_path, monkeypatch)
    source = tmp_path / project["id"] / "source" / "clip.mp4"
    source.write_bytes(b"fake-video-content")

    response = TestClient(web_api.app).get(
        f"/api/projects/{project['id']}/media/source/clip.mp4",
        headers={"Range": "bytes=0-3"},
    )

    assert response.status_code == 206
    assert response.content == b"fake"
    assert response.headers["content-range"] == "bytes 0-3/18"


def test_get_project_media_rejects_paths_outside_source(tmp_path: Path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    project = _create_project(tmp_path, monkeypatch)
    outside = tmp_path / project["id"] / "private.mp4"
    outside.write_bytes(b"not-public")

    response = TestClient(web_api.app).get(f"/api/projects/{project['id']}/media/private.mp4")

    assert response.status_code == 404


def test_background_analysis_fails_closed_when_runtime_capabilities_are_missing(
    tmp_path: Path, monkeypatch
) -> None:
    project = _create_project(tmp_path, monkeypatch)
    called = False

    def unavailable(_path):
        class Report:
            def require(self, *_names):
                raise RuntimeError("Required runtime capability unavailable: inspect_media")

        return Report()

    def unexpected_understand(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("analysis must not start without required capabilities")

    monkeypatch.setattr(web_api, "probe_runtime_capabilities", unavailable)
    monkeypatch.setattr(web_api, "understand_folder", unexpected_understand)

    web_api._run_project_analysis(project["id"])

    saved = web_api.get_project(project["id"])
    assert saved["analysis_status"] == "failed"
    assert "inspect_media" in saved["analysis_error"]
    assert called is False
