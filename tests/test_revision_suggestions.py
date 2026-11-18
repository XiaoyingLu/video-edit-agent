import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from travel_video_agent.cli import app
from travel_video_agent.editing.revision_review import (
    RevisionSuggestionReviewError,
    review_draft_revision_suggestions,
)
from travel_video_agent.editing.revision_suggestions import (
    build_draft_revision_suggestions,
    save_draft_revision_suggestions,
)
from travel_video_agent.editing.revisions import TimelineIntervalEdit
from travel_video_agent.editing.timeline_schemas import DraftTimelineClip, ProjectDraftTimeline


def _inputs(root: Path) -> tuple[Path, Path]:
    artifact = root / ".video-agent"
    artifact.mkdir(parents=True)
    timeline_path = artifact / "optimized-timeline.json"
    clips = [
        DraftTimelineClip(
            clip_id="shot-a",
            sequence_index=1,
            video_path="source.mp4",
            source_start_seconds=0,
            source_end_seconds=5,
            duration_seconds=5,
            combined_score=90,
            pacing_fit="within",
            pacing_target_seconds=4,
            pacing_note="Within target.",
            decision="approved",
        ),
        DraftTimelineClip(
            clip_id="shot-b",
            sequence_index=2,
            video_path="source.mp4",
            source_start_seconds=5,
            source_end_seconds=10,
            duration_seconds=5,
            combined_score=80,
            pacing_fit="long",
            pacing_target_seconds=3,
            pacing_note="Longer than target.",
            decision="approved",
        ),
    ]
    timeline = ProjectDraftTimeline(
        project_path=str(root.resolve()),
        style="cinematic",
        pacing="moderate",
        target_duration_seconds=10,
        proposed_duration_seconds=10,
        approved_duration_seconds=10,
        proposed_clip_count=2,
        clips=clips,
        timeline_approved=True,
    )
    timeline_path.write_text(timeline.model_dump_json(), encoding="utf-8")
    (root / "source.mp4").write_bytes(b"source")
    critique_path = artifact / "draft-critique.json"
    critique_path.write_text(
        json.dumps(
            {
                "project_path": str(root.resolve()),
                "source_timeline": ".video-agent/optimized-timeline.json",
                "semantic_reviews": [
                    {
                        "shot_id": "shot-a",
                        "timestamp_seconds": 2.5,
                        "match": "mismatch",
                        "score": 0.2,
                        "confidence": 0.9,
                        "evidence": "The frame shows food rather than mountains.",
                        "suggestion": "Review the footage assignment.",
                    },
                    {
                        "shot_id": "shot-b",
                        "timestamp_seconds": 7.5,
                        "match": "partial",
                        "score": 0.5,
                        "confidence": 0.4,
                        "evidence": "Low confidence.",
                        "suggestion": "Inspect.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return critique_path, timeline_path


def test_builds_confident_story_and_pacing_suggestions(tmp_path, monkeypatch):
    critique_path, timeline_path = _inputs(tmp_path)
    monkeypatch.setattr(
        "travel_video_agent.editing.revision_suggestions.validate_artifact",
        lambda *_: True,
    )

    report = build_draft_revision_suggestions(tmp_path)

    assert report.source_critique == critique_path.relative_to(tmp_path).as_posix()
    assert report.source_timeline == timeline_path.relative_to(tmp_path).as_posix()
    assert [(item.clip_id, item.category) for item in report.suggestions] == [
        ("shot-a", "story"),
        ("shot-b", "pacing"),
    ]
    story = report.suggestions[0]
    assert story.action == "review_footage"
    assert story.priority == "high"
    assert story.timestamp_seconds == 2.5
    assert story.status == "pending"
    assert report.suggestions[1].action == "review_timing"
    assert report.review_required is True
    assert report.timeline_modified is False
    assert report.footage_modified is False


def test_ignores_uncertain_semantic_reviews_and_unmapped_shots(tmp_path, monkeypatch):
    critique_path, _ = _inputs(tmp_path)
    payload = json.loads(critique_path.read_text())
    payload["semantic_reviews"].extend(
        [
            {
                "shot_id": "shot-a",
                "match": "mismatch",
                "confidence": 0.2,
                "timestamp_seconds": 1,
            },
            {
                "shot_id": "missing-shot",
                "match": "mismatch",
                "confidence": 0.99,
                "timestamp_seconds": 1,
            },
        ]
    )
    critique_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "travel_video_agent.editing.revision_suggestions.validate_artifact",
        lambda *_: True,
    )

    report = build_draft_revision_suggestions(tmp_path)

    assert [item.suggestion_id for item in report.suggestions] == [
        "shot-a-story",
        "shot-b-pacing",
    ]


def test_saves_suggestions_with_artifact_dependencies(tmp_path, monkeypatch):
    _inputs(tmp_path)
    monkeypatch.setattr(
        "travel_video_agent.editing.revision_suggestions.validate_artifact",
        lambda *_: True,
    )
    registered = {}
    monkeypatch.setattr(
        "travel_video_agent.editing.revision_suggestions.register_artifact",
        lambda root, path, dependencies: registered.update(
            root=root, path=path, dependencies=dependencies
        ),
    )

    report = build_draft_revision_suggestions(tmp_path)
    destination = save_draft_revision_suggestions(tmp_path, report)

    assert destination.is_file()
    assert json.loads(destination.read_text())["suggestions"][0]["clip_id"] == "shot-a"
    assert registered["path"] == destination
    assert len(registered["dependencies"]) == 2


def test_review_records_decisions_and_creates_separate_timeline_revision(
    tmp_path, monkeypatch
):
    _inputs(tmp_path)
    monkeypatch.setattr(
        "travel_video_agent.editing.revision_suggestions.validate_artifact",
        lambda *_: True,
    )
    monkeypatch.setattr(
        "travel_video_agent.editing.revision_review.validate_artifact",
        lambda *_: True,
    )
    monkeypatch.setattr(
        "travel_video_agent.editing.revisions.validate_artifact",
        lambda *_: True,
    )
    monkeypatch.setattr(
        "travel_video_agent.editing.revision_suggestions.register_artifact",
        lambda *_args, **_kwargs: None,
    )
    registered = []
    monkeypatch.setattr(
        "travel_video_agent.editing.revision_review.register_artifact",
        lambda root, path, dependencies: registered.append((path, dependencies)),
    )
    monkeypatch.setattr(
        "travel_video_agent.editing.revisions.register_artifact",
        lambda *_args, **_kwargs: None,
    )
    suggestions = build_draft_revision_suggestions(tmp_path)
    suggestion_file = save_draft_revision_suggestions(tmp_path, suggestions)

    result, review_file, revision = review_draft_revision_suggestions(
        tmp_path,
        suggestions_path=suggestion_file,
        accept_ids={"shot-b-pacing"},
        reject_ids={"shot-a-story"},
        interval_edits=[
            TimelineIntervalEdit(
                clip_id="shot-b",
                source_start_seconds=6,
                source_end_seconds=9,
            )
        ],
    )

    assert result.timeline_revision_path == revision.revision_path
    assert result.source_timeline_modified is False
    assert result.timeline_revision_created is True
    assert {item.suggestion_id: item.status for item in result.decisions} == {
        "shot-a-story": "rejected",
        "shot-b-pacing": "accepted",
    }
    assert review_file.is_file()
    assert revision.timeline.clips[1].source_start_seconds == 6
    original = json.loads((tmp_path / suggestions.source_timeline).read_text())
    assert original["timeline_approved"] is True
    assert original["clips"][1]["source_start_seconds"] == 5
    assert registered[0][0] == review_file


def test_review_rejects_interval_for_unaccepted_suggestion(tmp_path, monkeypatch):
    _inputs(tmp_path)
    monkeypatch.setattr(
        "travel_video_agent.editing.revision_suggestions.validate_artifact",
        lambda *_: True,
    )
    monkeypatch.setattr(
        "travel_video_agent.editing.revision_suggestions.register_artifact",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "travel_video_agent.editing.revision_review.validate_artifact",
        lambda *_: True,
    )
    suggestions = build_draft_revision_suggestions(tmp_path)
    suggestion_file = save_draft_revision_suggestions(tmp_path, suggestions)

    with pytest.raises(RevisionSuggestionReviewError, match="requires an accepted timing"):
        review_draft_revision_suggestions(
            tmp_path,
            suggestions_path=suggestion_file,
            accept_ids=set(),
            reject_ids=set(),
            interval_edits=[
                TimelineIntervalEdit(
                    clip_id="shot-b",
                    source_start_seconds=6,
                    source_end_seconds=9,
                )
            ],
        )


def test_review_revision_suggestions_cli_accepts_explicit_interval(
    tmp_path, monkeypatch
):
    _inputs(tmp_path)
    monkeypatch.setattr(
        "travel_video_agent.editing.revision_suggestions.validate_artifact",
        lambda *_: True,
    )
    monkeypatch.setattr(
        "travel_video_agent.editing.revision_suggestions.register_artifact",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "travel_video_agent.editing.revision_review.validate_artifact",
        lambda *_: True,
    )
    monkeypatch.setattr(
        "travel_video_agent.editing.revisions.validate_artifact",
        lambda *_: True,
    )
    monkeypatch.setattr(
        "travel_video_agent.editing.revision_review.register_artifact",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "travel_video_agent.editing.revisions.register_artifact",
        lambda *_args, **_kwargs: None,
    )
    report = build_draft_revision_suggestions(tmp_path)
    save_draft_revision_suggestions(tmp_path, report)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "review-revision-suggestions",
            str(tmp_path),
            "--accept",
            "shot-b-pacing",
            "--reject",
            "shot-a-story",
            "--set-interval",
            "shot-b:6:9",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "revision-001.json" in result.output
    assert (tmp_path / ".video-agent" / "timeline-revisions" / "revision-001.json").is_file()
