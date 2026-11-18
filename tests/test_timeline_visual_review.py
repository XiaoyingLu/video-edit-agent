import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from travel_video_agent.cli import app
from travel_video_agent.editing.timeline_schemas import DraftTimelineClip, ProjectDraftTimeline
from travel_video_agent.editing.timeline_visual_review import (
    TimelineVisualReviewError,
    build_timeline_visual_review,
    load_timeline_visual_review_decisions,
    render_timeline_visual_review_html,
    save_timeline_visual_review,
)


def _timeline(root: Path) -> Path:
    directory = root / ".video-agent"
    directory.mkdir(parents=True)
    timeline = ProjectDraftTimeline(
        project_path=str(root.resolve()),
        style="cinematic",
        pacing="moderate",
        target_duration_seconds=10,
        proposed_duration_seconds=4,
        approved_duration_seconds=0,
        proposed_clip_count=1,
        clips=[
            DraftTimelineClip(
                clip_id="coast-shot",
                sequence_index=1,
                video_path="coast.mp4",
                source_start_seconds=2,
                source_end_seconds=6,
                duration_seconds=4,
                combined_score=90,
                scene_summary="<sunset coastline>",
                temporal_summary="Waves roll toward shore",
                continuity_note="Opening shot",
                sequence_reason="Establishes the destination",
                pacing_note="A measured opening",
                decision="pending",
            )
        ],
    )
    path = directory / "optimized-timeline.json"
    path.write_text(timeline.model_dump_json(), encoding="utf-8")
    (root / "coast.mp4").write_bytes(b"fake video")
    return path


def _patch(monkeypatch):
    monkeypatch.setattr(
        "travel_video_agent.editing.timeline_visual_review.validate_artifact", lambda *_: True
    )
    monkeypatch.setattr(
        "travel_video_agent.editing.timeline_visual_review.register_artifact",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "travel_video_agent.editing.timeline_visual_review.shutil.which",
        lambda _name: "/usr/bin/ffmpeg",
    )

    def fake_run(command, timeout):
        Path(command[-1]).write_bytes(b"jpeg")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("travel_video_agent.editing.timeline_visual_review._run", fake_run)


def test_build_visual_review_extracts_midpoint_frame_without_mutating_timeline(
    tmp_path, monkeypatch
):
    source = _timeline(tmp_path)
    _patch(monkeypatch)

    report = build_timeline_visual_review(tmp_path)

    assert report.timeline_path == ".video-agent/optimized-timeline.json"
    assert report.timeline_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert len(report.clips) == 1
    assert report.clips[0].frame_data_uri.startswith("data:image/jpeg;base64,")
    assert report.clips[0].scene_summary == "<sunset coastline>"
    assert report.timeline_modified is False
    assert ProjectDraftTimeline.model_validate_json(source.read_text()).timeline_approved is False


def test_visual_review_rejects_stale_or_external_timeline(tmp_path, monkeypatch):
    source = _timeline(tmp_path)
    monkeypatch.setattr(
        "travel_video_agent.editing.timeline_visual_review.validate_artifact", lambda *_: False
    )
    with pytest.raises(TimelineVisualReviewError, match="stale or untracked"):
        build_timeline_visual_review(tmp_path)

    with pytest.raises(TimelineVisualReviewError, match="inside the project"):
        build_timeline_visual_review(tmp_path, timeline_path=tmp_path.parent / "outside.json")


def test_html_embeds_frames_and_exports_version_bound_decisions(tmp_path, monkeypatch):
    _timeline(tmp_path)
    _patch(monkeypatch)
    report = build_timeline_visual_review(tmp_path)

    rendered = render_timeline_visual_review_html(report)

    assert "data:image/jpeg;base64," in rendered
    assert "&lt;sunset coastline&gt;" in rendered
    assert "Export review decisions" in rendered
    assert report.timeline_sha256 in rendered
    assert "review-timeline --decisions-file" not in rendered


def test_save_review_writes_html_and_metadata_without_overwrite(tmp_path, monkeypatch):
    _timeline(tmp_path)
    _patch(monkeypatch)
    report = build_timeline_visual_review(tmp_path)

    json_file, html_file = save_timeline_visual_review(tmp_path, report)

    assert json_file.is_file()
    assert html_file.is_file()
    assert json.loads(json_file.read_text())["timeline_sha256"] == report.timeline_sha256
    with pytest.raises(FileExistsError):
        save_timeline_visual_review(tmp_path, report)


def test_decision_export_rejects_unsupported_values(tmp_path):
    path = tmp_path / "decisions.json"
    path.write_text(
        json.dumps({
            "schema_version": "1.0",
            "timeline_path": ".video-agent/optimized-timeline.json",
            "timeline_sha256": "abc",
            "decisions": {"shot-1": "maybe"},
        }),
        encoding="utf-8",
    )

    with pytest.raises(TimelineVisualReviewError, match="Unsupported decision"):
        load_timeline_visual_review_decisions(path)


def test_review_timeline_applies_browser_decisions_only_to_matching_timeline(
    tmp_path, monkeypatch
):
    from travel_video_agent.editing.timeline_visual_review import TimelineVisualReviewDecisions

    source = _timeline(tmp_path)
    timeline = ProjectDraftTimeline.model_validate_json(source.read_text())
    decision_file = tmp_path / "decisions.json"
    decision_file.write_text(
        TimelineVisualReviewDecisions(
            timeline_path=".video-agent/optimized-timeline.json",
            timeline_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            decisions={"coast-shot": "approved"},
        ).model_dump_json(),
        encoding="utf-8",
    )
    saved = []
    monkeypatch.setattr(
        "travel_video_agent.cli.load_draft_timeline", lambda *_: (timeline, source)
    )
    monkeypatch.setattr(
        "travel_video_agent.cli.save_draft_timeline", lambda result, path: saved.append(result)
    )

    result = CliRunner().invoke(
        app,
        [
            "review-timeline",
            str(tmp_path),
            "--decisions-file",
            str(decision_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert saved[0].timeline_approved is True
    assert saved[0].clips[0].decision == "approved"


def test_review_timeline_rejects_decisions_for_changed_timeline(tmp_path, monkeypatch):
    source = _timeline(tmp_path)
    timeline = ProjectDraftTimeline.model_validate_json(source.read_text())
    decision_file = tmp_path / "decisions.json"
    decision_file.write_text(
        json.dumps({
            "schema_version": "1.0",
            "timeline_path": ".video-agent/optimized-timeline.json",
            "timeline_sha256": "old-hash",
            "decisions": {"coast-shot": "approved"},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "travel_video_agent.cli.load_draft_timeline", lambda *_: (timeline, source)
    )

    result = CliRunner().invoke(
        app,
        [
            "review-timeline",
            str(tmp_path),
            "--decisions-file",
            str(decision_file),
        ],
    )

    assert result.exit_code == 1
    assert "Timeline changed after visual review" in result.output
