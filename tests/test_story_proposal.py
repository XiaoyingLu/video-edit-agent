from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from travel_video_agent.models import MediaInfo, ProjectAnalysis
from travel_video_agent.story_ledger import build_story_ledger
from travel_video_agent.story_proposal import (
    StoryProposal,
    StoryProposalError,
    generate_story_angles,
    propose_story,
    select_story_angle,
)
from travel_video_agent.understanding.schemas import (
    FrameObservation,
    SceneDynamics,
    VideoScene,
    VideoUnderstanding,
)


def _key(path: str) -> str:
    return hashlib.sha256(path.encode()).hexdigest()[:12]


def _scene(path: str, index: int, start: float, **overrides) -> VideoScene:
    values = {"setting": "mountain lake", "subject": "canoe", "mood": "serene"}
    values.update(overrides)
    return VideoScene(
        scene_id=f"{_key(path)}-scene-{index:04d}",
        start_seconds=start,
        end_seconds=start + 10,
        summary="A scenic lake.",
        story_function="establishing",
        dominant_mood=values["mood"],
        observations=[
            FrameObservation(
                timestamp_seconds=start + 1,
                frame_path="frame.jpg",
                description="A lake.",
                shot_scale="wide",
                setting=values["setting"],
                primary_subject=values["subject"],
            )
        ],
        dynamics=SceneDynamics(subject_motion="stationary", camera_motion="pan"),
    )


def _project(root: Path) -> list[str]:
    paths = ["day1.mp4", "day2.mp4"]
    media = []
    for path in paths:
        (root / path).write_bytes(b"video")
        os.utime(root / path, (1_790_000_000, 1_790_000_000))
        media.append(MediaInfo(path=path, duration_seconds=60, has_video=True, has_audio=True))
    os.utime(root / "day2.mp4", (1_790_086_400, 1_790_086_400))
    analysis = ProjectAnalysis(
        project_path=str(root), video_count=2, total_duration_seconds=120, videos=media
    )
    scenes = [
        _scene("day1.mp4", 1, 0),
        _scene("day1.mp4", 2, 10),
        _scene("day2.mp4", 1, 0, setting="summit trail", subject="hiker", mood="joyful"),
        _scene("day2.mp4", 2, 10, setting="summit trail", subject="hiker", mood="joyful"),
    ]
    understanding = VideoUnderstanding(
        project_path=str(root), video_count=2, scene_count=4, model="test", scenes=scenes
    )
    artifact_dir = root / ".video-agent"
    artifact_dir.mkdir()
    (artifact_dir / "analysis.json").write_text(analysis.model_dump_json(), encoding="utf-8")
    (artifact_dir / "understanding.json").write_text(
        understanding.model_dump_json(), encoding="utf-8"
    )
    return [scene.scene_id for scene in scenes]


class FakeStoryModel:
    model = "fake-story-model"

    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def generate_json(self, system_prompt: str, user_content: str, schema: dict) -> dict:
        self.calls.append((system_prompt, user_content))
        return self.response


def _angle(angle_id: str, chapters: list[dict]) -> dict:
    return {
        "angle_id": angle_id,
        "title": "Two Days Out",
        "logline": "A calm lake day, then a summit push.",
        "story_angle": "chronological journey",
        "chapters": chapters,
    }


def _chapter(chapter_id: str, seconds: float, scene_ids: list[str], **extra) -> dict:
    return {
        "chapter_id": chapter_id,
        "name": f"Chapter {chapter_id}",
        "purpose": "Move the story forward.",
        "target_seconds": seconds,
        "candidate_scene_ids": scene_ids,
        **extra,
    }


def test_grounding_drops_hallucinated_ids_and_fits_duration(tmp_path: Path) -> None:
    ids = _project(tmp_path)
    response = {
        "angles": [
            _angle(
                "x",
                [
                    _chapter(
                        "ch-1",
                        30,
                        [ids[0], "made-up-scene", ids[1]],
                        must_include_scene_ids=[ids[0]],
                    ),
                    _chapter("ch-2", 30, [ids[2], ids[3], ids[0]]),
                ],
            )
        ]
    }
    analysis = ProjectAnalysis.model_validate_json(
        (tmp_path / ".video-agent" / "analysis.json").read_text()
    )
    understanding = VideoUnderstanding.model_validate_json(
        (tmp_path / ".video-agent" / "understanding.json").read_text()
    )
    ledger = build_story_ledger(tmp_path, understanding, analysis)

    angles = generate_story_angles(ledger, FakeStoryModel(response), target_seconds=60)

    angle = angles[0]
    assert angle.angle_id == "angle-1"
    assert angle.chapters[0].candidate_scene_ids == [ids[0], ids[1]]
    assert angle.chapters[1].candidate_scene_ids == [ids[2], ids[3]]
    assert sum(chapter.target_seconds for chapter in angle.chapters) == pytest.approx(60)
    assert any("made-up-scene" in warning for warning in angle.warnings)
    assert any("already used" in warning for warning in angle.warnings)


def test_warns_when_candidate_footage_is_shorter_than_target(tmp_path: Path) -> None:
    ids = _project(tmp_path)
    response = {
        "angles": [_angle("a", [_chapter("ch-1", 10, [ids[0]]), _chapter("ch-2", 50, [ids[2]])])]
    }
    analysis = ProjectAnalysis.model_validate_json(
        (tmp_path / ".video-agent" / "analysis.json").read_text()
    )
    understanding = VideoUnderstanding.model_validate_json(
        (tmp_path / ".video-agent" / "understanding.json").read_text()
    )
    ledger = build_story_ledger(tmp_path, understanding, analysis)

    angle = generate_story_angles(ledger, FakeStoryModel(response), target_seconds=60)[0]

    assert any("ch-2" in warning and "less than" in warning for warning in angle.warnings)


def test_angle_with_only_unknown_scenes_is_skipped_and_all_skipped_fails(tmp_path: Path) -> None:
    ids = _project(tmp_path)
    analysis = ProjectAnalysis.model_validate_json(
        (tmp_path / ".video-agent" / "analysis.json").read_text()
    )
    understanding = VideoUnderstanding.model_validate_json(
        (tmp_path / ".video-agent" / "understanding.json").read_text()
    )
    ledger = build_story_ledger(tmp_path, understanding, analysis)
    bad = _angle("a", [_chapter("ch-1", 30, ["nope"]), _chapter("ch-2", 30, ["nope-2"])])
    good = _angle("b", [_chapter("ch-1", 30, [ids[0]]), _chapter("ch-2", 30, [ids[2]])])

    angles = generate_story_angles(
        ledger, FakeStoryModel({"angles": [bad, good]}), target_seconds=60, angle_count=2
    )
    assert [angle.angle_id for angle in angles] == ["angle-2"]

    with pytest.raises(StoryProposalError, match="No usable story angle"):
        generate_story_angles(ledger, FakeStoryModel({"angles": [bad]}), target_seconds=60)


def test_invalid_model_output_raises(tmp_path: Path) -> None:
    _project(tmp_path)
    analysis = ProjectAnalysis.model_validate_json(
        (tmp_path / ".video-agent" / "analysis.json").read_text()
    )
    understanding = VideoUnderstanding.model_validate_json(
        (tmp_path / ".video-agent" / "understanding.json").read_text()
    )
    ledger = build_story_ledger(tmp_path, understanding, analysis)

    with pytest.raises(StoryProposalError, match="invalid story proposal"):
        generate_story_angles(ledger, FakeStoryModel({"angles": "nonsense"}), target_seconds=60)


def test_propose_and_select_round_trip(tmp_path: Path) -> None:
    ids = _project(tmp_path)
    response = {
        "angles": [
            _angle(
                "a",
                [_chapter("ch-1", 45, [ids[0], ids[1]]), _chapter("ch-2", 45, [ids[2], ids[3]])],
            ),
            _angle("b", [_chapter("ch-1", 45, [ids[3]]), _chapter("ch-2", 45, [ids[0]])]),
        ]
    }
    model = FakeStoryModel(response)

    proposal, path = propose_story(
        tmp_path, prompt="a relaxed 90-second trip film", story_model=model
    )

    assert path == tmp_path / ".video-agent" / "story-proposal.json"
    assert proposal.target_duration_seconds == 90
    assert proposal.ledger_scene_count == 4
    assert proposal.selected_angle_id is None
    assert ids[0] in model.calls[0][1]  # the ledger text reached the model
    assert "relaxed 90-second trip film" in model.calls[0][1]

    selected, _ = select_story_angle(tmp_path, "angle-2")
    assert selected.selected_angle_id == "angle-2"
    reloaded = StoryProposal.model_validate(json.loads(path.read_text()))
    assert reloaded.selected_angle_id == "angle-2"

    with pytest.raises(ValueError, match="Unknown angle"):
        select_story_angle(tmp_path, "angle-9")


def test_propose_requires_understanding(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="analyze"):
        propose_story(tmp_path, story_model=FakeStoryModel({}))
