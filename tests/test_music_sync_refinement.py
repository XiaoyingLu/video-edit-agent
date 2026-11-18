"""Tests for reviewable music-sync timeline refinement."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from travel_video_agent.artifact_tracking import register_artifact
from travel_video_agent.editing.timeline_schemas import DraftTimelineClip, ProjectDraftTimeline
from travel_video_agent.models import MediaInfo
from travel_video_agent.music.refinement import (
    MusicSyncReviewDecision,
    MusicSyncReviewResult,
    apply_music_sync_review,
)
from travel_video_agent.music.sync import MusicSyncBoundary, MusicSyncSuggestions


def test_review_decision_records_accepted_rejected_and_infeasible_states() -> None:
    for status in ("accepted", "rejected", "skipped", "infeasible"):
        assert (
            MusicSyncReviewDecision(boundary_id="shot-001->shot-002", status=status).status
            == status
        )


def test_review_result_records_source_and_revision_provenance() -> None:
    result = MusicSyncReviewResult(
        source_timeline_path=".video-agent/optimized-timeline.json",
        suggestions_path=".video-agent/music-sync-suggestions.json",
        revision_path=".video-agent/music-sync-revisions/revision-001.json",
        decisions=[],
    )
    assert result.review_required is True


def test_review_decision_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        MusicSyncReviewDecision(boundary_id="shot-001->shot-002", status="auto-approved")


def _inputs(
    root: Path,
    *,
    transition: str = "cut",
    delta: float = 0.5,
    ranges: list[tuple[float, float]] | None = None,
) -> tuple[str, list[DraftTimelineClip]]:
    artifacts = root / ".video-agent"
    artifacts.mkdir(parents=True)
    (root / "source.mp4").write_bytes(b"source")
    ranges = ranges or [(1, 5), (5, 9)]
    clips = [
        DraftTimelineClip(
            clip_id=f"shot-{i:03}",
            sequence_index=i,
            video_path="source.mp4",
            source_start_seconds=start,
            source_end_seconds=end,
            duration_seconds=end - start,
            combined_score=100,
            pacing_note="test",
            transition_in="cut" if i == 1 else transition,
            transition_duration_seconds=0 if i == 1 or transition == "cut" else 0.5,
        )
        for i, (start, end) in enumerate(ranges, 1)
    ]
    total = sum(c.duration_seconds for c in clips) - sum(
        c.transition_duration_seconds for c in clips[1:]
    )
    timeline = artifacts / "optimized-timeline.json"
    timeline.write_text(
        ProjectDraftTimeline(
            project_path=str(root.resolve()),
            style="cinematic",
            pacing="moderate",
            target_duration_seconds=total,
            proposed_duration_seconds=total,
            proposed_clip_count=len(clips),
            clips=clips,
        ).model_dump_json()
    )
    register_artifact(root, timeline)
    boundaries = [
        MusicSyncBoundary(
            previous_clip_id=a.clip_id,
            next_clip_id=b.clip_id,
            boundary_seconds=a.duration_seconds,
            transition=b.transition_in,
            transition_duration_seconds=b.transition_duration_seconds,
            recommended_anchor="beat",
            recommended_boundary_seconds=a.duration_seconds + delta,
            adjustment_seconds=delta,
        )
        for a, b in zip(clips, clips[1:], strict=False)
    ]
    suggestions = artifacts / "music-sync-suggestions.json"
    suggestions.write_text(
        MusicSyncSuggestions(
            project_path=str(root.resolve()),
            timeline_path=timeline.relative_to(root).as_posix(),
            music_analysis_path=".video-agent/music-analysis.json",
            music_path=str(root / "track.wav"),
            beat_confidence=0.9,
            tolerance_seconds=0.2,
            boundary_count=len(boundaries),
            aligned_boundary_count=len(boundaries),
            boundaries=boundaries,
        ).model_dump_json()
    )
    register_artifact(root, suggestions, dependencies=[timeline])
    return "shot-001->shot-002", clips


def _probe(monkeypatch: pytest.MonkeyPatch, duration: float = 12) -> None:
    monkeypatch.setattr(
        "travel_video_agent.music.refinement.probe_video",
        lambda path, root: MediaInfo(
            path=path.name, duration_seconds=duration, has_video=True, has_audio=True
        ),
    )


def test_accepting_hard_cut_recommendation_moves_boundary_and_preserves_total_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key, original = _inputs(tmp_path)
    _probe(monkeypatch)
    result = apply_music_sync_review(tmp_path, accept_ids={key}, reject_ids=set())
    assert result.timeline.proposed_duration_seconds == 8
    assert (
        result.timeline.clips[0].source_end_seconds,
        result.timeline.clips[1].source_start_seconds,
    ) == (5.5, 5.5)
    assert sum(c.duration_seconds for c in result.timeline.clips) == sum(
        c.duration_seconds for c in original
    )


@pytest.mark.parametrize("transition", ["crossfade", "dissolve"])
def test_accepting_transition_recommendation_uses_transition_midpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transition: str
) -> None:
    key, _ = _inputs(tmp_path, transition=transition, delta=-0.25)
    _probe(monkeypatch)
    result = apply_music_sync_review(tmp_path, accept_ids={key}, reject_ids=set())
    assert (
        result.timeline.clips[0].source_end_seconds,
        result.timeline.clips[1].source_start_seconds,
    ) == (4.75, 4.75)
    assert result.timeline.clips[1].transition_in == transition


def test_adjustment_never_exceeds_source_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key, original = _inputs(tmp_path)
    _probe(monkeypatch, 5)
    result = apply_music_sync_review(tmp_path, accept_ids={key}, reject_ids=set())
    assert result.decisions[0].status == "infeasible" and result.timeline.clips == original


def test_adjustment_preserves_minimum_clip_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key, original = _inputs(tmp_path, ranges=[(1, 5), (5, 6)])
    _probe(monkeypatch)
    result = apply_music_sync_review(tmp_path, accept_ids={key}, reject_ids=set())
    assert result.decisions[0].status == "infeasible" and result.timeline.clips == original


def test_infeasible_adjustment_leaves_both_clips_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key, original = _inputs(tmp_path, delta=2)
    _probe(monkeypatch)
    result = apply_music_sync_review(
        tmp_path, accept_ids={key}, reject_ids=set(), minimum_clip_duration_seconds=3
    )
    assert result.decisions[0].status == "infeasible" and result.timeline.clips == original


def test_conflicting_adjacent_recommendations_are_processed_sequentially(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key, _ = _inputs(tmp_path, ranges=[(1, 5), (5, 6), (6, 10)])
    _probe(monkeypatch)
    result = apply_music_sync_review(
        tmp_path, accept_ids={key, "shot-002->shot-003"}, reject_ids=set()
    )
    assert [d.status for d in result.decisions] == ["infeasible", "accepted"]
    assert result.timeline.clips[1].source_end_seconds == 6.5


def test_revised_timeline_recalculates_proposed_duration_and_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key, _ = _inputs(tmp_path, transition="crossfade")
    _probe(monkeypatch)
    result = apply_music_sync_review(tmp_path, accept_ids={key}, reject_ids=set())
    assert (
        ProjectDraftTimeline.model_validate(
            json.loads(result.timeline.model_dump_json())
        ).proposed_duration_seconds
        == 7.5
    )
    assert result.timeline.timeline_approved is False


def test_review_creates_revision_001_without_overwriting_original_timeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key, _ = _inputs(tmp_path)
    original = (tmp_path / ".video-agent" / "optimized-timeline.json").read_bytes()
    _probe(monkeypatch)

    result = apply_music_sync_review(tmp_path, accept_ids={key}, reject_ids=set())

    revision = tmp_path / result.revision_path
    assert revision.name == "revision-001.json"
    assert revision.is_file()
    assert (tmp_path / ".video-agent" / "optimized-timeline.json").read_bytes() == original


def test_second_review_creates_revision_002_and_preserves_revision_001(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key, _ = _inputs(tmp_path)
    _probe(monkeypatch)
    first = apply_music_sync_review(tmp_path, accept_ids={key}, reject_ids=set())
    first_bytes = (tmp_path / first.revision_path).read_bytes()

    second = apply_music_sync_review(tmp_path, accept_ids=set(), reject_ids={key})

    assert second.revision_path.endswith("revision-002.json")
    assert (tmp_path / first.revision_path).read_bytes() == first_bytes


def test_review_record_persists_all_decision_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key, _ = _inputs(tmp_path, ranges=[(1, 5), (5, 6), (6, 10)])
    _probe(monkeypatch)
    result = apply_music_sync_review(
        tmp_path, accept_ids={key, "shot-002->shot-003"}, reject_ids=set()
    )

    review = tmp_path / result.revision_path.replace(".json", ".review.json")
    assert [item["status"] for item in json.loads(review.read_text())["decisions"]] == [
        "infeasible",
        "accepted",
    ]


def test_revision_is_registered_with_timeline_and_suggestion_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key, _ = _inputs(tmp_path)
    _probe(monkeypatch)
    result = apply_music_sync_review(tmp_path, accept_ids={key}, reject_ids=set())
    manifest = json.loads((tmp_path / ".video-agent" / "artifact-manifest.json").read_text())

    assert manifest["artifacts"][result.revision_path]["dependencies"] == [
        ".video-agent/music-sync-suggestions.json",
        ".video-agent/optimized-timeline.json",
    ]


def test_stale_suggestions_are_rejected_without_creating_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key, _ = _inputs(tmp_path)
    suggestions = tmp_path / ".video-agent" / "music-sync-suggestions.json"
    suggestions.write_text(suggestions.read_text() + " ")
    _probe(monkeypatch)

    with pytest.raises(ValueError):
        apply_music_sync_review(tmp_path, accept_ids={key}, reject_ids=set())

    assert not (tmp_path / ".video-agent" / "music-sync-revisions").exists()


def test_missing_source_video_fails_without_creating_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key, _ = _inputs(tmp_path)
    (tmp_path / "source.mp4").unlink()
    _probe(monkeypatch)

    with pytest.raises(ValueError):
        apply_music_sync_review(tmp_path, accept_ids={key}, reject_ids=set())

    assert not (tmp_path / ".video-agent" / "music-sync-revisions").exists()
