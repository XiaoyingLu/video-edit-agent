from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from travel_video_agent.cli import app
from travel_video_agent.editing.revision_visual_review import (
    build_timeline_revision_visual_review,
    render_timeline_revision_review_html,
    save_timeline_revision_visual_review,
)
from travel_video_agent.editing.revisions import TimelineIntervalEdit, create_timeline_revision
from travel_video_agent.editing.timeline_schemas import DraftTimelineClip, ProjectDraftTimeline


def _approved_timeline(root: Path) -> Path:
    artifact = root / ".video-agent"
    artifact.mkdir(parents=True, exist_ok=True)
    clips = [
        DraftTimelineClip(
            clip_id="opening",
            sequence_index=1,
            video_path="trip.mp4",
            source_start_seconds=1,
            source_end_seconds=5,
            duration_seconds=4,
            combined_score=90,
            pacing_note="Opening shot.",
            decision="approved",
        ),
        DraftTimelineClip(
            clip_id="coast",
            sequence_index=2,
            video_path="trip.mp4",
            source_start_seconds=8,
            source_end_seconds=12,
            duration_seconds=4,
            combined_score=85,
            transition_in="crossfade",
            transition_duration_seconds=0.8,
            pacing_note="Coastal shot.",
            decision="approved",
        ),
    ]
    timeline = ProjectDraftTimeline(
        project_path=str(root.resolve()),
        style="vlog",
        pacing="moderate",
        target_duration_seconds=12,
        proposed_duration_seconds=7.2,
        approved_duration_seconds=7.2,
        proposed_clip_count=2,
        clips=clips,
        timeline_approved=True,
    )
    source = artifact / "optimized-timeline.json"
    source.write_text(timeline.model_dump_json(), encoding="utf-8")
    (root / "trip.mp4").write_bytes(b"fake source video")
    return source


def _patch_tools(monkeypatch):
    monkeypatch.setattr(
        "travel_video_agent.editing.revisions.validate_artifact", lambda *_: True
    )
    monkeypatch.setattr(
        "travel_video_agent.editing.revisions.register_artifact",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "travel_video_agent.editing.revision_visual_review.validate_artifact",
        lambda *_: True,
    )
    monkeypatch.setattr(
        "travel_video_agent.editing.revision_visual_review.shutil.which",
        lambda name: f"/usr/bin/{name}",
    )

    def fake_run(command, timeout):
        Path(command[-1]).write_bytes(b"fake jpeg")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(
        "travel_video_agent.editing.revision_visual_review._run", fake_run
    )


def _revision(root, monkeypatch):
    source = _approved_timeline(root)
    _patch_tools(monkeypatch)
    result = create_timeline_revision(
        root,
        [
            TimelineIntervalEdit(
                clip_id="coast",
                source_start_seconds=8.5,
                source_end_seconds=11.5,
            )
        ],
    )
    return source, root / result.revision_path


def test_visual_review_pairs_midpoint_source_frames_and_is_read_only(tmp_path, monkeypatch):
    source, revision = _revision(tmp_path, monkeypatch)

    report = build_timeline_revision_visual_review(
        tmp_path,
        before_path=source,
        after_path=revision,
    )

    assert len(report.frames) == 1
    frame = report.frames[0]
    assert frame.clip_id == "coast"
    assert frame.before_start_seconds == 8
    assert frame.before_end_seconds == 12
    assert frame.after_start_seconds == 8.5
    assert frame.after_end_seconds == 11.5
    assert frame.before_frame_data_uri.startswith("data:image/jpeg;base64,")
    assert frame.after_frame_data_uri.startswith("data:image/jpeg;base64,")
    assert report.duration_delta_seconds < 0
    assert "not rendered playback" in render_timeline_revision_review_html(report)
    original = ProjectDraftTimeline.model_validate_json(source.read_text())
    revised = ProjectDraftTimeline.model_validate_json(revision.read_text())
    assert original.timeline_approved is True
    assert revised.timeline_approved is False


def test_visual_review_saves_self_contained_html_and_json(tmp_path, monkeypatch):
    source, revision = _revision(tmp_path, monkeypatch)
    report = build_timeline_revision_visual_review(
        tmp_path,
        before_path=source,
        after_path=revision,
    )
    registered = []
    monkeypatch.setattr(
        "travel_video_agent.editing.revision_visual_review.register_artifact",
        lambda root, path, dependencies: registered.append((path, dependencies)),
    )

    json_file, html_file = save_timeline_revision_visual_review(
        tmp_path,
        report,
        output_path=Path(".video-agent/revision-review/review-001"),
    )

    assert json_file.is_file()
    assert html_file.is_file()
    assert "data:image/jpeg;base64," in html_file.read_text()
    assert "before_frame_data_uri" not in json_file.read_text()
    assert len(registered) == 2


def test_visual_review_cli_writes_report(tmp_path, monkeypatch):
    source, revision = _revision(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "travel_video_agent.editing.revision_visual_review.register_artifact",
        lambda *_args, **_kwargs: None,
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "review-timeline-revision",
            str(tmp_path),
            "--before",
            str(source),
            "--after",
            str(revision),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Visual review:" in result.output
    assert (tmp_path / ".video-agent" / "timeline-revision-review" / "review.html").is_file()
