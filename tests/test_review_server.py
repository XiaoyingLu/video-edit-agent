import hashlib
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from video_edit_agent.editing.review_server import create_review_server
from video_edit_agent.editing.timeline_schemas import DraftTimelineClip, ProjectDraftTimeline


def _post(server, payload, *, token=None):
    host, port = server.server_address
    request = urllib.request.Request(
        f"http://{host}:{port}/__video_agent__/timeline-edits",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Origin": f"http://{host}:{port}",
            "X-Video-Agent-Token": token or server.session_token,
        },
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=3)


def _start_server(tmp_path):
    server = create_review_server(tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_server(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=3)


def test_review_server_injects_save_control_into_timeline_page(tmp_path: Path):
    page = tmp_path / ".video-agent" / "timeline-visual-review" / "review.html"
    page.parent.mkdir(parents=True)
    page.write_text(
        "<html><head><title>Visual Timeline Review</title></head>"
        "<body><header>Review</header></body></html>",
        encoding="utf-8",
    )
    server, thread = _start_server(tmp_path)
    try:
        host, port = server.server_address
        assert host == "127.0.0.1"
        with urllib.request.urlopen(
            f"http://{host}:{port}/.video-agent/timeline-visual-review/review.html",
            timeout=3,
        ) as response:
            content = response.read().decode()
            headers = response.headers
        assert "Save as draft revision" in content
        assert "/__video_agent__/timeline-edits" in content
        assert headers["Cache-Control"] == "no-store"
    finally:
        _stop_server(server, thread)


def test_review_server_rejects_missing_or_invalid_session_token(tmp_path: Path):
    server, thread = _start_server(tmp_path)
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            _post(server, {}, token="invalid")
        assert error.value.code == 403
    finally:
        _stop_server(server, thread)


def test_review_server_rejects_requests_from_other_origins(tmp_path: Path):
    server, thread = _start_server(tmp_path)
    try:
        host, port = server.server_address
        request = urllib.request.Request(
            f"http://{host}:{port}/__video_agent__/timeline-edits",
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "Origin": "http://example.com",
                "X-Video-Agent-Token": server.session_token,
            },
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=3)
        assert error.value.code == 403
    finally:
        _stop_server(server, thread)


def test_review_server_saves_a_separate_unapproved_timeline_revision(tmp_path, monkeypatch):
    artifact_dir = tmp_path / ".video-agent"
    artifact_dir.mkdir()
    source = ProjectDraftTimeline(
        project_path=str(tmp_path.resolve()),
        style="cinematic",
        pacing="moderate",
        target_duration_seconds=10,
        proposed_duration_seconds=8,
        approved_duration_seconds=0,
        proposed_clip_count=2,
        clips=[
            DraftTimelineClip(
                clip_id="coast",
                sequence_index=1,
                video_path="coast.mp4",
                source_start_seconds=2,
                source_end_seconds=6,
                duration_seconds=4,
                combined_score=90,
                pacing_note="Opening",
            ),
            DraftTimelineClip(
                clip_id="mountain",
                sequence_index=2,
                video_path="mountain.mp4",
                source_start_seconds=10,
                source_end_seconds=14,
                duration_seconds=4,
                combined_score=85,
                pacing_note="Follow-up",
            ),
        ],
    )
    timeline = artifact_dir / "optimized-timeline.json"
    timeline.write_text(source.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        "video_edit_agent.editing.timeline_visual_editor.validate_artifact",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        "video_edit_agent.editing.timeline_visual_editor.register_artifact",
        lambda *_args, **_kwargs: None,
    )
    payload = {
        "schema_version": "1.0",
        "timeline_path": ".video-agent/optimized-timeline.json",
        "timeline_sha256": hashlib.sha256(timeline.read_bytes()).hexdigest(),
        "clips": [
            {
                "clip_id": "mountain",
                "source_start_seconds": 10.5,
                "source_end_seconds": 13.5,
            },
            {
                "clip_id": "coast",
                "source_start_seconds": 2.5,
                "source_end_seconds": 5.5,
            },
        ],
    }
    server, thread = _start_server(tmp_path)
    try:
        with _post(server, payload) as response:
            result = json.loads(response.read())
        revision_path = tmp_path / result["revision_path"]
        revision = ProjectDraftTimeline.model_validate_json(revision_path.read_text())
        assert revision_path.is_file()
        assert [clip.clip_id for clip in revision.clips] == ["mountain", "coast"]
        assert revision.timeline_approved is False
        assert revision.approved_duration_seconds == 0
        assert [
            clip.clip_id
            for clip in ProjectDraftTimeline.model_validate_json(timeline.read_text()).clips
        ] == ["coast", "mountain"]
    finally:
        _stop_server(server, thread)


def test_review_server_does_not_serve_environment_files(tmp_path: Path):
    secret = tmp_path / ".env"
    secret.write_text("TOKEN=do-not-serve", encoding="utf-8")
    server, thread = _start_server(tmp_path)
    try:
        host, port = server.server_address
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(f"http://{host}:{port}/.env", timeout=3)
        assert error.value.code == 404
    finally:
        _stop_server(server, thread)


def test_review_server_can_bind_all_container_interfaces(tmp_path: Path):
    server = create_review_server(tmp_path, host="0.0.0.0")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert server.server_address[0] == "0.0.0.0"
        _, port = server.server_address
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=3)
        assert error.value.code == 404
    finally:
        _stop_server(server, thread)


def test_review_server_rejects_unsafe_bind_addresses(tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported bind host"):
        create_review_server(tmp_path, host="192.168.1.10")


def test_live_review_saves_clip_decisions_without_approving_timeline(tmp_path, monkeypatch):
    from video_edit_agent.editing.timeline_schemas import DraftTimelineClip, ProjectDraftTimeline
    artifact_dir=tmp_path/".video-agent"; artifact_dir.mkdir()
    timeline=ProjectDraftTimeline(project_path=str(tmp_path.resolve()),style="cinematic",pacing="moderate",
      target_duration_seconds=10,proposed_duration_seconds=4,proposed_clip_count=1,
      clips=[DraftTimelineClip(clip_id="coast",sequence_index=1,video_path="coast.mp4",
        source_start_seconds=0,source_end_seconds=4,duration_seconds=4,combined_score=90,pacing_note="Opening")])
    path=artifact_dir/"optimized-timeline.json"; path.write_text(timeline.model_dump_json(),encoding="utf-8")
    monkeypatch.setattr("video_edit_agent.editing.review_server.validate_artifact",lambda *_:True)
    payload={"schema_version":"1.0","timeline_path":".video-agent/optimized-timeline.json",
      "timeline_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"decisions":{"coast":"approved"},
      "approve_timeline":False}
    server,thread=_start_server(tmp_path)
    try:
      host,port=server.server_address
      request=urllib.request.Request(f"http://{host}:{port}/__video_agent__/timeline-review",
        data=json.dumps(payload).encode(),headers={"Content-Type":"application/json","Origin":f"http://{host}:{port}",
        "X-Video-Agent-Token":server.session_token},method="POST")
      with urllib.request.urlopen(request,timeout=3) as response: result=json.loads(response.read())
      saved=ProjectDraftTimeline.model_validate_json(path.read_text())
      assert result["approved_count"]==1 and saved.clips[0].decision=="approved"
      assert saved.timeline_approved is False
    finally: _stop_server(server,thread)


def test_live_review_rejects_timeline_approval_with_pending_clips(tmp_path, monkeypatch):
    from video_edit_agent.editing.timeline_schemas import DraftTimelineClip, ProjectDraftTimeline
    artifact_dir=tmp_path/".video-agent"; artifact_dir.mkdir()
    timeline=ProjectDraftTimeline(project_path=str(tmp_path.resolve()),style="cinematic",pacing="moderate",
      target_duration_seconds=10,proposed_duration_seconds=4,proposed_clip_count=1,
      clips=[DraftTimelineClip(clip_id="coast",sequence_index=1,video_path="coast.mp4",
        source_start_seconds=0,source_end_seconds=4,duration_seconds=4,combined_score=90,pacing_note="Opening")])
    path=artifact_dir/"optimized-timeline.json"; path.write_text(timeline.model_dump_json(),encoding="utf-8")
    monkeypatch.setattr("video_edit_agent.editing.review_server.validate_artifact",lambda *_:True)
    payload={"schema_version":"1.0","timeline_path":".video-agent/optimized-timeline.json",
      "timeline_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"decisions":{"coast":"pending"},
      "approve_timeline":True}
    server,thread=_start_server(tmp_path)
    try:
      host,port=server.server_address
      request=urllib.request.Request(f"http://{host}:{port}/__video_agent__/timeline-review",
        data=json.dumps(payload).encode(),headers={"Content-Type":"application/json","Origin":f"http://{host}:{port}",
        "X-Video-Agent-Token":server.session_token},method="POST")
      with pytest.raises(urllib.error.HTTPError) as error: urllib.request.urlopen(request,timeout=3)
      assert error.value.code==400
      assert ProjectDraftTimeline.model_validate_json(path.read_text()).timeline_approved is False
    finally: _stop_server(server,thread)
