from __future__ import annotations

import hashlib
import os
from datetime import datetime
from pathlib import Path

from travel_video_agent.highlights.schemas import (
    HighlightCandidate,
    HighlightSignals,
    ProjectHighlights,
)
from travel_video_agent.models import MediaInfo, ProjectAnalysis
from travel_video_agent.story_ledger import build_story_ledger
from travel_video_agent.transcription.schemas import (
    ProjectTranscription,
    TranscriptSegment,
    VideoTranscript,
)
from travel_video_agent.understanding.schemas import (
    FrameObservation,
    SceneDynamics,
    VideoScene,
    VideoUnderstanding,
)


def _video_key(path: str) -> str:
    return hashlib.sha256(path.encode()).hexdigest()[:12]


def _scene(
    path: str,
    index: int,
    start: float,
    *,
    setting: str = "rocky coastline",
    subject: str = "ocean",
    mood: str = "serene",
    scale: str = "wide",
) -> VideoScene:
    return VideoScene(
        scene_id=f"{_video_key(path)}-scene-{index:04d}",
        start_seconds=start,
        end_seconds=start + 5,
        summary="A scenic coastline.",
        story_function="establishing",
        dominant_mood=mood,
        observations=[
            FrameObservation(
                timestamp_seconds=start + 1,
                frame_path="frame.jpg",
                description="A coastline.",
                shot_scale=scale,
                setting=setting,
                primary_subject=subject,
            )
        ],
        dynamics=SceneDynamics(
            subject_motion="stationary",
            camera_motion="pan",
        ),
    )


def _inputs(root: Path, paths: list[str]):
    media = []
    for path in paths:
        source = root / path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"video")
        media.append(
            MediaInfo(
                path=path,
                duration_seconds=30,
                has_video=True,
                has_audio=True,
            )
        )
    analysis = ProjectAnalysis(
        project_path=str(root),
        video_count=len(media),
        total_duration_seconds=sum(item.duration_seconds for item in media),
        videos=media,
    )
    return analysis


def test_story_ledger_joins_visual_speech_and_highlight_context(tmp_path: Path) -> None:
    analysis = _inputs(tmp_path, ["day1.mp4"])
    timestamp = 1_790_000_000
    os.utime(tmp_path / "day1.mp4", (timestamp, timestamp))
    understanding = VideoUnderstanding(
        project_path=str(tmp_path),
        video_count=1,
        scene_count=1,
        model="test",
        scenes=[_scene("day1.mp4", 1, 0)],
    )
    transcription = ProjectTranscription(
        project_path=str(tmp_path),
        video_count=1,
        transcribed_video_count=1,
        segment_count=2,
        model="test",
        videos=[
            VideoTranscript(
                path="day1.mp4",
                duration_seconds=30,
                has_audio=True,
                segments=[
                    TranscriptSegment(start_seconds=1, end_seconds=2, text="  unreal! "),
                    TranscriptSegment(start_seconds=8, end_seconds=9, text="outside scene"),
                ],
            )
        ],
    )
    highlights = ProjectHighlights(
        project_path=str(tmp_path),
        video_count=1,
        candidate_count=1,
        candidates=[
            HighlightCandidate(
                candidate_id="candidate-1",
                video_path="day1.mp4",
                start_seconds=0,
                end_seconds=5,
                duration_seconds=5,
                score=81,
                signals=HighlightSignals(
                    motion=0.2, sharpness=0.9, exposure=0.8, saturation=0.5
                ),
            )
        ],
    )

    ledger = build_story_ledger(
        tmp_path,
        understanding,
        analysis,
        transcription=transcription,
        highlights=highlights,
    )

    entry = ledger.entries[0]
    assert entry.scene_id == understanding.scenes[0].scene_id
    assert entry.setting == "rocky coastline"
    assert entry.subject == "ocean"
    assert entry.shot_scale == "wide"
    assert entry.motion == "stationary/pan"
    assert entry.highlight_score == 0.81
    assert entry.speech == ("unreal!",)
    assert "highlight 0.81" in entry.render()
    assert '"unreal!"' in entry.render()
    assert len(ledger.clusters) == 1
    assert "1 scenes grouped into 1 clusters" in ledger.render()


def test_story_ledger_clusters_related_scenes_and_separates_days(tmp_path: Path) -> None:
    analysis = _inputs(tmp_path, ["morning.mp4", "next-day.mp4"])
    first = 1_790_000_000
    second = first + 86_400
    os.utime(tmp_path / "morning.mp4", (first, first))
    os.utime(tmp_path / "next-day.mp4", (second, second))
    scenes = [
        _scene("morning.mp4", 1, 0),
        _scene("morning.mp4", 2, 6),
        _scene("next-day.mp4", 1, 0, setting="mountain trail", subject="hiker"),
    ]
    understanding = VideoUnderstanding(
        project_path=str(tmp_path),
        video_count=2,
        scene_count=3,
        model="test",
        scenes=scenes,
    )

    ledger = build_story_ledger(tmp_path, understanding, analysis)

    assert len(ledger.entries) == 3
    assert [entry.day_label for entry in ledger.entries] == ["day1", "day1", "day2"]
    assert len(ledger.clusters) == 2
    assert ledger.clusters[0].scene_ids == [scenes[0].scene_id, scenes[1].scene_id]
    assert ledger.clusters[1].scene_ids == [scenes[2].scene_id]


def test_story_ledger_uses_scene_ids_and_ignores_non_overlapping_speech(tmp_path: Path) -> None:
    analysis = _inputs(tmp_path, ["clip.mp4"])
    understanding = VideoUnderstanding(
        project_path=str(tmp_path),
        video_count=1,
        scene_count=1,
        model="test",
        scenes=[_scene("clip.mp4", 1, 10)],
    )
    transcription = ProjectTranscription(
        project_path=str(tmp_path),
        video_count=1,
        transcribed_video_count=1,
        segment_count=1,
        model="test",
        videos=[
            VideoTranscript(
                path="clip.mp4",
                duration_seconds=30,
                has_audio=True,
                segments=[TranscriptSegment(start_seconds=0, end_seconds=2, text="not here")],
            )
        ],
    )

    ledger = build_story_ledger(tmp_path, understanding, analysis, transcription=transcription)

    assert ledger.entries[0].speech == ()
    assert ledger.entries[0].scene_id == understanding.scenes[0].scene_id


def test_story_ledger_rejects_scenes_without_analyzed_source_video(tmp_path: Path) -> None:
    analysis = _inputs(tmp_path, ["clip.mp4"])
    understanding = VideoUnderstanding(
        project_path=str(tmp_path),
        video_count=1,
        scene_count=1,
        model="test",
        scenes=[
            VideoScene(
                scene_id="unknownvideo-scene-0001",
                start_seconds=0,
                end_seconds=5,
                summary="Unknown source.",
            )
        ],
    )

    import pytest

    with pytest.raises(ValueError, match="absent from analysis"):
        build_story_ledger(tmp_path, understanding, analysis)



def test_story_ledger_prefers_embedded_capture_timestamp(tmp_path: Path) -> None:
    source_path = "camera.mp4"
    analysis = _inputs(tmp_path, [source_path])
    analysis.videos[0].capture_timestamp = "2026-06-12T09:30:00Z"
    os.utime(tmp_path / source_path, (1_600_000_000, 1_600_000_000))
    scene = _scene(source_path, 1, 12)
    understanding = VideoUnderstanding(
        project_path=str(tmp_path),
        video_count=1,
        scene_count=1,
        model="test",
        scenes=[scene],
    )

    ledger = build_story_ledger(tmp_path, understanding, analysis)

    expected = datetime.fromisoformat("2026-06-12T09:30:00+00:00").astimezone()
    assert ledger.entries[0].day_label == "day1"
    assert ledger.entries[0].timestamp_label == (
        datetime.fromtimestamp(expected.timestamp() + 12).astimezone().strftime("%H:%M")
    )
    assert "embedded media creation timestamps" in ledger.timestamp_note


def test_story_ledger_caps_clusters_without_dropping_or_reordering_scenes(
    tmp_path: Path,
) -> None:
    analysis = _inputs(tmp_path, ["long-day.mp4"])
    scenes = [
        _scene(
            "long-day.mp4",
            index,
            float(index * 6),
            setting=f"location {index}",
            subject=f"subject {index}",
        )
        for index in range(1, 26)
    ]
    understanding = VideoUnderstanding(
        project_path=str(tmp_path),
        video_count=1,
        scene_count=len(scenes),
        model="test",
        scenes=scenes,
    )

    ledger = build_story_ledger(tmp_path, understanding, analysis)

    assert len(ledger.clusters) == 20
    flattened_ids = [
        scene_id for cluster in ledger.clusters for scene_id in cluster.scene_ids
    ]
    assert flattened_ids == [entry.scene_id for entry in ledger.entries]
    assert len(flattened_ids) == len(scenes)
