import json

import pytest
from typer.testing import CliRunner

from travel_video_agent.cli import app
from travel_video_agent.editing.revisions import (
    TimelineIntervalEdit,
    TimelineRevisionError,
    compare_timeline_files,
    create_timeline_revision,
)
from travel_video_agent.editing.timeline_pipeline import apply_timeline_review, build_draft_timeline

from tests.test_timeline_optimization import _write_inputs

runner = CliRunner()


def _approved_timeline(root):
    _write_inputs(root)
    timeline = build_draft_timeline(root)
    timeline = apply_timeline_review(timeline, {"earlier", "later"}, set())
    source = root / ".video-agent" / "optimized-timeline.json"
    source.write_text(timeline.model_dump_json(), encoding="utf-8")
    return source, timeline


def _patch_artifact_tracking(monkeypatch):
    monkeypatch.setattr("travel_video_agent.editing.revisions.validate_artifact", lambda *_: True)
    monkeypatch.setattr("travel_video_agent.editing.revisions.register_artifact", lambda *_args, **_kwargs: None)


def test_create_revision_trims_only_selected_clip_and_resets_its_approval(tmp_path, monkeypatch):
    source, timeline = _approved_timeline(tmp_path)
    _patch_artifact_tracking(monkeypatch)

    result = create_timeline_revision(
        tmp_path,
        [TimelineIntervalEdit(clip_id="later", source_start_seconds=8.5, source_end_seconds=11.5)],
    )

    revised = result.timeline
    earlier, later = revised.clips
    assert result.source_timeline_path == source.relative_to(tmp_path).as_posix()
    assert result.changed_clip_ids == ["later"]
    assert earlier.decision == "approved"
    assert later.decision == "pending"
    assert later.source_start_seconds == 8.5
    assert later.source_end_seconds == 11.5
    assert later.duration_seconds == 3
    assert later.transition_duration_seconds == 0.75
    assert revised.timeline_approved is False
    assert revised.approved_duration_seconds == 4
    assert result.before_duration_seconds == 7.2
    assert result.after_duration_seconds == 6.25
    assert (tmp_path / result.revision_path).is_file()
    assert (tmp_path / result.revision_path).with_suffix(".review.json").is_file()

    original = json.loads(source.read_text(encoding="utf-8"))
    assert original["timeline_approved"] is True
    assert original["clips"][1]["source_start_seconds"] == 8


def test_revision_rejects_extension_outside_existing_source_interval(tmp_path, monkeypatch):
    _approved_timeline(tmp_path)
    _patch_artifact_tracking(monkeypatch)

    with pytest.raises(TimelineRevisionError, match="existing source interval"):
        create_timeline_revision(
            tmp_path,
            [TimelineIntervalEdit(clip_id="later", source_start_seconds=7.5, source_end_seconds=11)],
        )


def test_revision_rejects_duplicate_and_unknown_clip_edits(tmp_path, monkeypatch):
    _approved_timeline(tmp_path)
    _patch_artifact_tracking(monkeypatch)

    edit = TimelineIntervalEdit(clip_id="later", source_start_seconds=8.5, source_end_seconds=11.5)
    with pytest.raises(TimelineRevisionError, match="duplicate"):
        create_timeline_revision(tmp_path, [edit, edit])
    with pytest.raises(TimelineRevisionError, match="unknown clip"):
        create_timeline_revision(
            tmp_path,
            [TimelineIntervalEdit(clip_id="missing", source_start_seconds=0, source_end_seconds=2)],
        )


def test_compare_timeline_files_reports_interval_and_duration_delta(tmp_path, monkeypatch):
    source, _ = _approved_timeline(tmp_path)
    _patch_artifact_tracking(monkeypatch)
    result = create_timeline_revision(
        tmp_path,
        [TimelineIntervalEdit(clip_id="later", source_start_seconds=8.5, source_end_seconds=11.5)],
    )
    comparison = compare_timeline_files(tmp_path, source, tmp_path / result.revision_path)

    assert len(comparison.changes) == 1
    assert comparison.changes[0].clip_id == "later"
    assert comparison.changes[0].before_start_seconds == 8
    assert comparison.changes[0].after_start_seconds == 8.5
    assert comparison.duration_delta_seconds == -0.95


def test_revision_cli_and_timeline_comparison_cli(tmp_path, monkeypatch):
    source, _ = _approved_timeline(tmp_path)
    _patch_artifact_tracking(monkeypatch)

    created = runner.invoke(
        app,
        [
            "revise-timeline",
            str(tmp_path),
            "--set-interval",
            "later:8.5:11.5",
        ],
    )
    assert created.exit_code == 0, created.output
    revision = tmp_path / ".video-agent" / "timeline-revisions" / "revision-001.json"
    assert revision.is_file()

    compared = runner.invoke(
        app,
        [
            "compare-timelines",
            str(tmp_path),
            "--before",
            str(source),
            "--after",
            str(revision),
        ],
    )
    assert compared.exit_code == 0, compared.output
    assert "later" in compared.output
    assert "-0.95s" in compared.output
