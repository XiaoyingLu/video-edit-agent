import json
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
        output_aspect_ratio=None,
        output_resolution=None,
    ):
        calls.append(
            {
                "timeline_path": timeline_path,
                "output_path": output_path,
                "overwrite": overwrite,
                "dry_run": dry_run,
                "fit_mode": fit_mode,
                "output_aspect_ratio": output_aspect_ratio,
                "output_resolution": output_resolution,
            }
        )
        if dry_run:
            return None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"rendered preview")
        dimensions = {
            ("16:9", "720p"): (1280, 720),
            ("9:16", "720p"): (720, 1280),
            ("1:1", "720p"): (720, 720),
        }
        width, height = dimensions.get(
            (output_aspect_ratio, output_resolution), (1920, 1080)
        )
        return RenderReport(
            output_path=str(output_path),
            source_timeline=str(timeline_path),
            source_video_count=1,
            segment_count=1,
            expected_duration_seconds=3,
            actual_duration_seconds=3,
            width=width,
            height=height,
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


def _write_project_plan(root: Path, aspect_ratio: str = "16:9") -> None:
    plan_path = root / ".video-agent" / "video-project-plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(
            {
                "project_path": str(root.resolve()),
                "brief": {
                    "source_prompt": "Create a travel video",
                    "target_duration_seconds": 10,
                    "style": "vlog",
                    "aspect_ratio": aspect_ratio,
                    "resolution": "4k",
                    "pacing": "moderate",
                },
                "editing_preferences": {
                    "source_prompt": "Create a travel video",
                    "target_duration_seconds": 10,
                    "style": "vlog",
                    "aspect_ratio": aspect_ratio,
                    "resolution": "4k",
                    "pacing": "moderate",
                },
            }
        ),
        encoding="utf-8",
    )


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
    assert calls[0]["output_resolution"] is None


@pytest.mark.parametrize(
    ("aspect_ratio", "expected_dimensions"),
    [
        ("16:9", (1280, 720)),
        ("9:16", (720, 1280)),
        ("1:1", (720, 720)),
    ],
)
def test_revision_preview_uses_720p_and_preserves_project_aspect_ratio(
    tmp_path, monkeypatch, aspect_ratio, expected_dimensions
):
    revision = _approved_revision(tmp_path)
    _write_project_plan(tmp_path, aspect_ratio=aspect_ratio)
    calls = _mock_renderer(monkeypatch)

    report, manifest = render_timeline_revision_preview(tmp_path, timeline_path=revision)

    assert report is not None
    assert (report.width, report.height) == expected_dimensions
    assert calls[0]["output_aspect_ratio"] == aspect_ratio
    assert calls[0]["output_resolution"] == "720p"
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved["preview_resolution"] == "720p"


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
