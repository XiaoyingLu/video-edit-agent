from pathlib import Path

import pytest
from typer.testing import CliRunner

from video_edit_agent.cli import app
from video_edit_agent.editing.revision_preview import render_timeline_revision_preview
from video_edit_agent.editing.timeline_schemas import DraftTimelineClip, ProjectDraftTimeline
from video_edit_agent.rendering.renderer import RenderReport


def _approved_revision(root: Path) -> Path:
    revision_dir = root / ".video-agent" / "timeline-revisions"
    revision_dir.mkdir(parents=True)
    (root / "trip.mp4").write_bytes(b"source")
    clip = DraftTimelineClip(
        clip_id="shot-001",
        sequence_index=1,
        video_path="trip.mp4",
        source_start_seconds=1,
        source_end_seconds=4,
        duration_seconds=3,
        combined_score=90,
        pacing_note="Reviewed shot.",
        decision="approved",
    )
    timeline = ProjectDraftTimeline(
        project_path=str(root.resolve()),
        style="vlog",
        pacing="moderate",
        target_duration_seconds=3,
        proposed_duration_seconds=3,
        approved_duration_seconds=3,
        proposed_clip_count=1,
        clips=[clip],
        timeline_approved=True,
    )
    revision = revision_dir / "revision-001.json"
    revision.write_text(timeline.model_dump_json(), encoding="utf-8")
    return revision


def _mock_renderer(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "video_edit_agent.editing.revision_preview.validate_artifact",
        lambda *_: True,
    )

    def fake_render(
        folder,
        *,
        timeline_path,
        output_path,
        overwrite,
        timeout_seconds,
        dry_run,
        fit_mode,
        title_overlay_path,
    ):
        calls.append(
            {
                "timeline_path": timeline_path,
                "output_path": output_path,
                "overwrite": overwrite,
                "dry_run": dry_run,
                "fit_mode": fit_mode,
            }
        )
        if dry_run:
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"rendered preview")
        return RenderReport(
            output_path=str(output_path),
            source_timeline=str(timeline_path),
            source_video_count=1,
            segment_count=1,
            expected_duration_seconds=3,
            actual_duration_seconds=3,
            width=1920,
            height=1080,
            video_codec="h264",
            audio_codec="aac",
        )

    monkeypatch.setattr(
        "video_edit_agent.editing.revision_preview.render_approved_shot_timeline",
        fake_render,
    )
    monkeypatch.setattr(
        "video_edit_agent.editing.revision_preview.register_artifact",
        lambda *_args, **_kwargs: None,
    )
    return calls


def test_revision_preview_renders_separate_video_and_manifest(tmp_path, monkeypatch):
    revision = _approved_revision(tmp_path)
    calls = _mock_renderer(monkeypatch)

    report, manifest = render_timeline_revision_preview(
        tmp_path,
        timeline_path=revision,
    )

    preview = tmp_path / ".video-agent" / "revision-previews" / "revision-001.mp4"
    assert report is not None
    assert report.output_path == str(preview)
    assert preview.read_bytes() == b"rendered preview"
    assert manifest == preview.with_suffix(".preview.json")
    assert manifest.is_file()
    assert calls[0]["timeline_path"] == revision
    assert calls[0]["overwrite"] is False
    assert calls[0]["fit_mode"] == "crop"


def test_revision_preview_requires_approved_revision(tmp_path, monkeypatch):
    revision = _approved_revision(tmp_path)
    timeline = ProjectDraftTimeline.model_validate_json(revision.read_text())
    timeline.timeline_approved = False
    revision.write_text(timeline.model_dump_json(), encoding="utf-8")
    calls = _mock_renderer(monkeypatch)

    with pytest.raises(ValueError, match="not fully approved"):
        render_timeline_revision_preview(tmp_path, timeline_path=revision)

    assert calls == []


def test_revision_preview_rejects_non_revision_timeline(tmp_path, monkeypatch):
    _approved_revision(tmp_path)
    calls = _mock_renderer(monkeypatch)

    with pytest.raises(ValueError, match="timeline-revisions"):
        render_timeline_revision_preview(
            tmp_path,
            timeline_path=tmp_path / ".video-agent" / "optimized-timeline.json",
        )

    assert calls == []


def test_revision_preview_dry_run_does_not_write_manifest(tmp_path, monkeypatch):
    revision = _approved_revision(tmp_path)
    calls = _mock_renderer(monkeypatch)

    report, manifest = render_timeline_revision_preview(
        tmp_path,
        timeline_path=revision,
        dry_run=True,
    )

    assert report is None
    assert manifest is None
    assert calls[0]["dry_run"] is True
    assert not (tmp_path / ".video-agent" / "revision-previews").exists()


def test_revision_preview_refuses_existing_output(tmp_path, monkeypatch):
    revision = _approved_revision(tmp_path)
    calls = _mock_renderer(monkeypatch)
    preview = tmp_path / ".video-agent" / "revision-previews" / "revision-001.mp4"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="already exists"):
        render_timeline_revision_preview(tmp_path, timeline_path=revision)

    assert preview.read_bytes() == b"keep"
    assert calls == []


def test_revision_preview_cli_renders_and_reports_manifest(tmp_path, monkeypatch):
    revision = _approved_revision(tmp_path)
    _mock_renderer(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "render-revision-preview",
            str(tmp_path),
            "--timeline",
            str(revision),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Approved revision preview rendered" in result.output
    assert "Preview manifest:" in result.output
    assert (tmp_path / ".video-agent" / "revision-previews" / "revision-001.preview.json").is_file()
