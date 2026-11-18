"""Schema tests for reviewable music-sync timeline refinement."""

import pytest
from pydantic import ValidationError

from travel_video_agent.music.refinement import (
    MusicSyncReviewDecision,
    MusicSyncReviewResult,
)


def test_review_decision_records_accepted_rejected_and_infeasible_states() -> None:
    for status in ("accepted", "rejected", "skipped", "infeasible"):
        decision = MusicSyncReviewDecision(
            boundary_id="shot-001->shot-002",
            status=status,
            reason="Insufficient source handles." if status == "infeasible" else None,
        )
        assert decision.status == status


def test_review_result_records_source_and_revision_provenance() -> None:
    result = MusicSyncReviewResult(
        source_timeline_path=".video-agent/optimized-timeline.json",
        suggestions_path=".video-agent/music-sync-suggestions.json",
        revision_path=".video-agent/music-sync-revisions/revision-001.json",
        decisions=[
            MusicSyncReviewDecision(
                boundary_id="shot-001->shot-002",
                status="accepted",
            ),
            MusicSyncReviewDecision(
                boundary_id="shot-002->shot-003",
                status="skipped",
            ),
        ],
    )

    assert result.review_required is True
    assert result.source_timeline_path.endswith("optimized-timeline.json")
    assert result.suggestions_path.endswith("music-sync-suggestions.json")
    assert result.revision_path.endswith("revision-001.json")
    assert [item.status for item in result.decisions] == ["accepted", "skipped"]


def test_review_decision_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        MusicSyncReviewDecision(
            boundary_id="shot-001->shot-002",
            status="auto-approved",
        )
