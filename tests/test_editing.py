import json
from pathlib import Path

import pytest

from travel_video_agent.editing.pipeline import build_edit_proposal


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

    from travel_video_agent.understanding.schemas import VideoScene, VideoUnderstanding

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
        "videos": [{
            "path": video_path,
            "duration_seconds": 12,
            "has_audio": True,
            "transcript_available": True,
            "audio_silences": [],
            "candidates": [{
                "start_seconds": 3,
                "end_seconds": 5,
                "duration_seconds": 2,
            }],
        }],
    }
    understanding = VideoUnderstanding(
        project_path=str(tmp_path),
        video_count=1,
        scene_count=1,
        model="qwen2.5vl:3b",
        scenes=[VideoScene(
            scene_id=f"{key}-scene-0001",
            start_seconds=2,
            end_seconds=6,
            summary="A mountain lake at sunset",
        )],
    )
    (artifact_dir / "silence-analysis.json").write_text(json.dumps(silence), encoding="utf-8")
    (artifact_dir / "understanding.json").write_text(
        understanding.model_dump_json(), encoding="utf-8"
    )

    item = build_edit_proposal(tmp_path).videos[0].review_items[0]
    assert item.scene_id == f"{key}-scene-0001"
    assert item.scene_summary == "A mountain lake at sunset"
    assert item.decision == "pending"
