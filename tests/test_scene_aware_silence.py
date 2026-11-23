import hashlib
import json
from pathlib import Path

import pytest

from video_edit_agent.editing.pipeline import build_edit_proposal
from video_edit_agent.understanding.schemas import FrameObservation, VideoScene, VideoUnderstanding


def _write_analysis_artifacts(tmp_path: Path, scenes: list[VideoScene]) -> None:
    artifact_dir = tmp_path / ".video-agent"
    artifact_dir.mkdir()
    silence = {
        "project_path": str(tmp_path),
        "video_count": 1,
        "analyzed_video_count": 1,
        "noise_threshold_db": -35,
        "min_silence_duration": 1.5,
        "speech_padding_seconds": 0.25,
        "videos": [
            {
                "path": "clip.mp4",
                "duration_seconds": 10,
                "has_audio": True,
                "transcript_available": True,
                "audio_silences": [],
                "candidates": [
                    {
                        "start_seconds": 3,
                        "end_seconds": 7,
                        "duration_seconds": 4,
                        "review_required": True,
                        "review_reason": "review",
                    }
                ],
            }
        ],
    }
    (artifact_dir / "silence-analysis.json").write_text(json.dumps(silence), encoding="utf-8")
    understanding = VideoUnderstanding(
        project_path=str(tmp_path),
        video_count=1,
        scene_count=len(scenes),
        scenes=scenes,
        model="test-model",
    )
    (artifact_dir / "understanding.json").write_text(
        understanding.model_dump_json(), encoding="utf-8"
    )


def _scene_id(index: int) -> str:
    video_key = hashlib.sha256(b"clip.mp4").hexdigest()[:12]
    return f"{video_key}-scene-{index:04d}"


@pytest.mark.parametrize("role", ["scenic_breath", "establishing"])
def test_scenic_and_establishing_silence_is_marked_preserve(tmp_path: Path, role: str) -> None:
    scene = VideoScene(
        scene_id=_scene_id(1),
        start_seconds=2,
        end_seconds=8,
        summary="A quiet mountain panorama",
        story_function=role,
    )
    _write_analysis_artifacts(tmp_path, [scene])

    item = build_edit_proposal(tmp_path).videos[0].review_items[0]

    assert item.suggested_action == "preserve"
    assert item.scene_role == role
    assert item.scene_id == scene.scene_id
    assert item.decision == "pending"
    assert item.review_required is True
    assert "preserve this scenic/establishing moment by default" in item.review_reason


def test_any_overlapping_scenic_frame_takes_precedence_over_larger_action_scene(
    tmp_path: Path,
) -> None:
    action_scene = VideoScene(
        scene_id=_scene_id(1),
        start_seconds=2,
        end_seconds=6.5,
        summary="A person walking along a trail",
        story_function="action",
    )
    scenic_scene = VideoScene(
        scene_id=_scene_id(2),
        start_seconds=6.5,
        end_seconds=8,
        summary="A still alpine lake",
        story_function="unknown",
        observations=[
            FrameObservation(
                timestamp_seconds=7,
                frame_path=".video-agent/frames/lake.jpg",
                description="A still alpine lake",
                visual_role="scenic_breath",
            )
        ],
    )
    _write_analysis_artifacts(tmp_path, [action_scene, scenic_scene])

    item = build_edit_proposal(tmp_path).videos[0].review_items[0]

    assert item.suggested_action == "preserve"
    assert item.scene_role == "scenic_breath"
    assert item.scene_id == scenic_scene.scene_id
    assert item.scene_summary == "A still alpine lake"
    assert item.decision == "pending"


def test_action_scene_keeps_neutral_removal_suggestion(tmp_path: Path) -> None:
    scene = VideoScene(
        scene_id=_scene_id(1),
        start_seconds=2,
        end_seconds=8,
        summary="A hiker walking uphill",
        story_function="action",
    )
    _write_analysis_artifacts(tmp_path, [scene])

    item = build_edit_proposal(tmp_path).videos[0].review_items[0]

    assert item.suggested_action == "consider_removing"
    assert item.scene_role == "action"
    assert item.decision == "pending"
    assert "Suggested action: preserve" not in item.review_reason
