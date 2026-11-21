import json
from pathlib import Path

import pytest

from video_edit_agent.editing.pipeline import build_edit_proposal


def test_builds_review_only_items_from_silence_candidates(tmp_path: Path) -> None:
    artifact_dir = tmp_path / ".video-agent"
    artifact_dir.mkdir()
    report = {
        "project_path": str(tmp_path),
        "video_count": 1,
        "analyzed_video_count": 1,
        "noise_threshold_db": -35,
        "min_silence_duration": 1.5,
        "speech_padding_seconds": 0.25,
        "videos": [
            {
                "path": "clip.mp4",
                "duration_seconds": 12,
                "has_audio": True,
                "transcript_available": True,
                "audio_silences": [],
                "candidates": [
                    {
                        "start_seconds": 3,
                        "end_seconds": 5,
                        "duration_seconds": 2,
                        "review_required": True,
                        "review_reason": "review",
                    }
                ],
            }
        ],
    }
    (artifact_dir / "silence-analysis.json").write_text(json.dumps(report), encoding="utf-8")

    proposal = build_edit_proposal(tmp_path)

    assert proposal.review_item_count == 1
    item = proposal.videos[0].review_items[0]
    assert item.item_id == "clip-review-001"
    assert item.suggested_action == "consider_removing"
    assert item.decision == "pending"
    assert item.review_required is True
    assert proposal.approval_required is True
    assert proposal.rendering_performed is False


def test_missing_silence_report_has_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="detect-silence"):
        build_edit_proposal(tmp_path)


def test_scene_context_is_optional_and_never_auto_approves(tmp_path: Path) -> None:
    artifact_dir = tmp_path / ".video-agent"
    artifact_dir.mkdir()
    report = {
        "project_path": str(tmp_path),
        "video_count": 1,
        "analyzed_video_count": 1,
        "noise_threshold_db": -35,
        "min_silence_duration": 1.5,
        "speech_padding_seconds": 0.25,
        "videos": [
            {
                "path": "clip.mp4",
                "duration_seconds": 12,
                "has_audio": True,
                "transcript_available": True,
                "audio_silences": [],
                "candidates": [
                    {
                        "start_seconds": 3,
                        "end_seconds": 5,
                        "duration_seconds": 2,
                    }
                ],
            }
        ],
    }
    (artifact_dir / "silence-analysis.json").write_text(json.dumps(report), encoding="utf-8")
    proposal = build_edit_proposal(tmp_path)
    item = proposal.videos[0].review_items[0]
    assert item.scene_id is None
    assert item.decision == "pending"


def test_scene_context_is_joined_by_source_video_hash(tmp_path: Path) -> None:
    import hashlib

    from video_edit_agent.understanding.schemas import VideoScene, VideoUnderstanding

    artifact_dir = tmp_path / ".video-agent"
    artifact_dir.mkdir()
    video_path = "clip.mp4"
    key = hashlib.sha256(video_path.encode("utf-8")).hexdigest()[:12]
    silence = {
        "project_path": str(tmp_path),
        "video_count": 1,
        "analyzed_video_count": 1,
        "noise_threshold_db": -35,
        "min_silence_duration": 1.5,
        "speech_padding_seconds": 0.25,
        "videos": [
            {
                "path": video_path,
                "duration_seconds": 12,
                "has_audio": True,
                "transcript_available": True,
                "audio_silences": [],
                "candidates": [
                    {
                        "start_seconds": 3,
                        "end_seconds": 5,
                        "duration_seconds": 2,
                    }
                ],
            }
        ],
    }
    understanding = VideoUnderstanding(
        project_path=str(tmp_path),
        video_count=1,
        scene_count=1,
        model="qwen2.5vl:3b",
        scenes=[
            VideoScene(
                scene_id=f"{key}-scene-0001",
                start_seconds=2,
                end_seconds=6,
                summary="A mountain lake at sunset",
            )
        ],
    )
    (artifact_dir / "silence-analysis.json").write_text(json.dumps(silence), encoding="utf-8")
    (artifact_dir / "understanding.json").write_text(
        understanding.model_dump_json(), encoding="utf-8"
    )

    item = build_edit_proposal(tmp_path).videos[0].review_items[0]
    assert item.scene_id == f"{key}-scene-0001"
    assert item.scene_summary == "A mountain lake at sunset"
    assert item.decision == "pending"


def test_review_item_rejects_invalid_interval() -> None:
    from pydantic import ValidationError

    from video_edit_agent.editing.schemas import EditReviewItem

    with pytest.raises(ValidationError):
        EditReviewItem(
            item_id="invalid",
            start_seconds=5,
            end_seconds=4,
            duration_seconds=1,
            review_reason="invalid bounds",
        )


def test_review_item_rejects_duration_mismatch() -> None:
    from pydantic import ValidationError

    from video_edit_agent.editing.schemas import EditReviewItem

    with pytest.raises(ValidationError):
        EditReviewItem(
            item_id="invalid",
            start_seconds=1,
            end_seconds=4,
            duration_seconds=1,
            review_reason="inconsistent duration",
        )


def _review_proposal(tmp_path: Path):
    from video_edit_agent.editing.schemas import (
        EditReviewItem,
        ProjectEditProposal,
        VideoEditProposal,
    )

    return ProjectEditProposal(
        project_path=str(tmp_path.resolve()),
        source_artifacts=[".video-agent/silence-analysis.json"],
        video_count=1,
        review_item_count=2,
        videos=[
            VideoEditProposal(
                path="clip.mp4",
                duration_seconds=10,
                review_items=[
                    EditReviewItem(
                        item_id="clip-review-001",
                        start_seconds=2,
                        end_seconds=4,
                        duration_seconds=2,
                        review_reason="Review first interval.",
                    ),
                    EditReviewItem(
                        item_id="clip-review-002",
                        start_seconds=6,
                        end_seconds=7,
                        duration_seconds=1,
                        review_reason="Review second interval.",
                    ),
                ],
            )
        ],
    )


def test_review_decisions_generate_retained_timeline(tmp_path: Path) -> None:
    from video_edit_agent.editing.pipeline import (
        apply_review_decisions,
        build_project_timeline,
    )

    proposal = _review_proposal(tmp_path)
    with pytest.raises(ValueError, match="all review items"):
        build_project_timeline(proposal)

    apply_review_decisions(
        proposal,
        approve_ids={"clip-review-001"},
        reject_ids={"clip-review-002"},
    )
    timeline = build_project_timeline(proposal)

    assert timeline.approved_cut_count == 1
    assert timeline.retained_duration_seconds == 8
    assert [
        (segment.start_seconds, segment.end_seconds) for segment in timeline.videos[0].segments
    ] == [(0, 2), (4, 10)]
    assert timeline.rendering_performed is False


def test_review_decisions_reject_unknown_and_conflicting_ids(tmp_path: Path) -> None:
    from video_edit_agent.editing.pipeline import apply_review_decisions

    proposal = _review_proposal(tmp_path)
    with pytest.raises(ValueError, match="unknown review item"):
        apply_review_decisions(proposal, {"missing"}, set())
    with pytest.raises(ValueError, match="both approved and rejected"):
        apply_review_decisions(proposal, {"clip-review-001"}, {"clip-review-001"})


def test_timeline_rejects_overlapping_approved_cuts(tmp_path: Path) -> None:
    from video_edit_agent.editing.pipeline import apply_review_decisions, build_project_timeline

    proposal = _review_proposal(tmp_path)
    proposal.videos[0].review_items[1].start_seconds = 3
    proposal.videos[0].review_items[1].end_seconds = 5
    proposal.videos[0].review_items[1].duration_seconds = 2
    apply_review_decisions(
        proposal,
        {"clip-review-001", "clip-review-002"},
        set(),
    )
    with pytest.raises(ValueError, match="overlap"):
        build_project_timeline(proposal)


def test_review_proposal_must_match_project_folder(tmp_path: Path) -> None:
    from video_edit_agent.editing.pipeline import load_edit_proposal, save_edit_proposal

    artifact_dir = tmp_path / ".video-agent"
    artifact_dir.mkdir()
    proposal = _review_proposal(tmp_path / "elsewhere")
    save_edit_proposal(proposal, artifact_dir / "edit-proposal.json")
    with pytest.raises(ValueError, match="different project"):
        load_edit_proposal(tmp_path)
