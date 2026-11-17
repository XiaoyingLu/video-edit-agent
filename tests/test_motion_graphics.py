import json
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from travel_video_agent.cli import app
from travel_video_agent.editing.timeline_schemas import ProjectDraftTimeline
from travel_video_agent.motion_graphics.pipeline import (
    MotionGraphicsError,
    _composition_html,
    create_hyperframes_project,
)


def _approved_timeline(root: Path) -> Path:
    artifact = root / ".video-agent" / "optimized-timeline.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "project_path": str(root),
        "style": "cinematic",
        "pacing": "slow",
        "target_duration_seconds": 5,
        "proposed_duration_seconds": 5,
        "approved_duration_seconds": 5,
        "proposed_clip_count": 1,
        "timeline_approved": True,
        "clips": [
            {
                "clip_id": "clip-1",
                "sequence_index": 1,
                "video_path": str(root / "source.mp4"),
                "source_start_seconds": 0,
                "source_end_seconds": 5,
                "duration_seconds": 5,
                "combined_score": 90,
                "pacing_note": "A scenic establishing shot.",
                "decision": "approved",
            }
        ],
    }
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    ProjectDraftTimeline.model_validate(payload)
    return artifact


def _fake_ffprobe(monkeypatch, has_audio=True):
    def run(command, **kwargs):
        streams = [{"codec_type": "video", "width": 1920, "height": 1080}]
        if has_audio:
            streams.append({"codec_type": "audio"})
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"format": {"duration": "5.0"}, "streams": streams}).encode(),
            stderr=b"",
        )
    monkeypatch.setattr("travel_video_agent.motion_graphics.pipeline.subprocess.run", run)


def test_composition_escapes_user_text_and_preserves_audio():
    html = _composition_html(
        duration=12,
        width=1920,
        height=1080,
        title="Nova Scotia <Coast>",
        location="Peggy’s Cove & Halifax",
        has_audio=True,
    )
    assert "Nova Scotia &lt;Coast&gt;" in html
    assert "Peggy’s Cove &amp; Halifax" in html
    assert 'data-composition-id="main"' in html
    assert 'data-duration="12.000"' in html
    assert "window.__timelines.main = timeline" in html
    assert 'data-has-audio="true"' in html
    assert "muted" not in html


def test_composition_mutes_video_without_audio():
    html = _composition_html(
        duration=5,
        width=1280,
        height=720,
        title="A Journey",
        location=None,
        has_audio=False,
    )
    assert "muted" in html
    assert "location-card" not in html


def test_generates_hyperframes_project_from_approved_draft(tmp_path, monkeypatch):
    timeline = _approved_timeline(tmp_path)
    video = tmp_path / ".video-agent" / "rendered-draft.mp4"
    video.write_bytes(b"test-video")
    _fake_ffprobe(monkeypatch)

    report = create_hyperframes_project(tmp_path, title="A Coastal Journey", location="Nova Scotia")

    project = Path(report.project_directory)
    assert (project / "index.html").is_file()
    assert (project / "meta.json").is_file()
    copied_media = project / "assets" / "rendered-draft.mp4"
    assert copied_media.is_file()
    assert copied_media.read_bytes() == video.read_bytes()
    assert copied_media.stat().st_ino != video.stat().st_ino
    assert report.width == 1920 and report.height == 1080
    assert report.duration_seconds == 5
    assert report.has_audio is True
    assert "Nova Scotia" in (project / "index.html").read_text()
    from travel_video_agent.artifact_tracking import validate_artifact

    assert validate_artifact(tmp_path, project / "index.html")
    assert timeline.is_file()


def test_rejects_unapproved_timeline(tmp_path, monkeypatch):
    timeline = _approved_timeline(tmp_path)
    payload = json.loads(timeline.read_text())
    payload["timeline_approved"] = False
    timeline.write_text(json.dumps(payload))
    video = tmp_path / ".video-agent" / "rendered-draft.mp4"
    video.write_bytes(b"test-video")
    _fake_ffprobe(monkeypatch)
    with pytest.raises(MotionGraphicsError, match="not approved"):
        create_hyperframes_project(tmp_path, title="A Journey")


def test_rejects_paths_outside_project(tmp_path):
    with pytest.raises(MotionGraphicsError, match="inside the project folder"):
        create_hyperframes_project(tmp_path, title="A Journey", video_path=Path("/tmp/outside.mp4"))


def test_existing_project_requires_overwrite(tmp_path, monkeypatch):
    _approved_timeline(tmp_path)
    video = tmp_path / ".video-agent" / "rendered-draft.mp4"
    video.write_bytes(b"test-video")
    _fake_ffprobe(monkeypatch)
    create_hyperframes_project(tmp_path, title="A Journey")
    with pytest.raises(FileExistsError, match="--overwrite"):
        create_hyperframes_project(tmp_path, title="A Journey")


def test_create_motion_project_cli_invokes_generator(tmp_path, monkeypatch):
    from travel_video_agent import cli

    expected = {
        "project_directory": str(tmp_path / ".video-agent" / "hyperframes-project"),
        "composition_path": str(tmp_path / ".video-agent" / "hyperframes-project" / "index.html"),
        "media_path": str(
            tmp_path / ".video-agent" / "hyperframes-project" / "assets" / "rendered-draft.mp4"
        ),
        "duration_seconds": 5.0,
        "width": 1920,
        "height": 1080,
        "has_audio": True,
        "title": "A Journey",
        "location": "Nova Scotia",
    }
    monkeypatch.setattr(
        cli, "create_hyperframes_project", lambda *args, **kwargs: type("Report", (), expected)()
    )
    result = CliRunner().invoke(
        app,
        [
            "create-motion-project",
            str(tmp_path),
            "--title",
            "A Journey",
            "--location",
            "Nova Scotia",
        ],
    )
    assert result.exit_code == 0
    assert "HyperFrames project generated" in result.stdout
    assert "npx hyperframes preview" in result.stdout
