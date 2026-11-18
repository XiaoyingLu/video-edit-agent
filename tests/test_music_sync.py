import json
from pathlib import Path

import pytest

from travel_video_agent.artifact_tracking import register_artifact
from travel_video_agent.editing.timeline_schemas import DraftTimelineClip, ProjectDraftTimeline
from travel_video_agent.music.schemas import MusicAnalysis
from travel_video_agent.music.sync import MusicSyncError, suggest_music_sync


def _write_inputs(
    root: Path,
    *,
    confidence: float = 0.9,
    beats: list[float] | None = None,
    downbeats: list[float] | None = None,
) -> tuple[Path, Path]:
    artifact_dir = root / ".video-agent"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    soundtrack = root / "track.wav"
    soundtrack.write_bytes(b"soundtrack placeholder")
    timeline = ProjectDraftTimeline(
        project_path=str(root.resolve()),
        style="cinematic",
        pacing="moderate",
        target_duration_seconds=7,
        proposed_duration_seconds=6.5,
        proposed_clip_count=2,
        clips=[
            DraftTimelineClip(
                clip_id="shot-001",
                sequence_index=1,
                video_path="source.mp4",
                source_start_seconds=0,
                source_end_seconds=4,
                duration_seconds=4,
                combined_score=100,
                pacing_note="Opening shot.",
                decision="pending",
            ),
            DraftTimelineClip(
                clip_id="shot-002",
                sequence_index=2,
                video_path="source.mp4",
                source_start_seconds=4,
                source_end_seconds=7,
                duration_seconds=3,
                combined_score=100,
                pacing_note="Second shot.",
                transition_in="crossfade",
                transition_duration_seconds=0.5,
                decision="pending",
            ),
        ],
    )
    timeline_path = artifact_dir / "optimized-timeline.json"
    timeline_path.write_text(timeline.model_dump_json(), encoding="utf-8")
    register_artifact(root, timeline_path)

    analysis = MusicAnalysis(
        project_path=str(root.resolve()),
        music_path=str(soundtrack.resolve()),
        duration_seconds=20,
        sample_rate=11025,
        bpm=120,
        beat_confidence=confidence,
        beat_interval_seconds=0.5,
        beat_times_seconds=beats if beats is not None else [3.7, 4.0, 4.3],
        downbeat_times_seconds=downbeats if downbeats is not None else [4.0],
        suggested_cut_points_seconds=beats if beats is not None else [3.7, 4.0, 4.3],
    )
    music_path = artifact_dir / "music-analysis.json"
    music_path.write_text(analysis.model_dump_json(), encoding="utf-8")
    register_artifact(root, music_path, dependencies=[soundtrack])
    return timeline_path, music_path


def test_music_sync_suggests_nearest_beat_without_changing_timeline(tmp_path: Path) -> None:
    timeline_path, music_path = _write_inputs(tmp_path)
    before = timeline_path.read_bytes()

    result = suggest_music_sync(tmp_path)

    assert result.boundary_count == 1
    assert result.aligned_boundary_count == 1
    boundary = result.boundaries[0]
    # Crossfade midpoint: 4.0s accumulated duration minus half of the 0.5s overlap.
    assert boundary.boundary_seconds == 3.75
    assert boundary.nearest_beat_seconds == 3.7
    assert boundary.beat_delta_seconds == -0.05
    assert boundary.recommended_anchor == "beat"
    assert boundary.recommended_boundary_seconds == 3.7
    assert result.timeline_modified is False
    assert timeline_path.read_bytes() == before
    assert timeline_path.is_file() and music_path.is_file()


def test_music_sync_prefers_downbeat_when_it_is_the_closest_anchor(tmp_path: Path) -> None:
    _write_inputs(tmp_path, beats=[3.7, 4.0], downbeats=[3.7])

    result = suggest_music_sync(tmp_path)

    assert result.boundaries[0].recommended_anchor == "downbeat"
    assert result.boundaries[0].recommended_boundary_seconds == 3.7


def test_music_sync_does_not_recommend_when_confidence_is_low(tmp_path: Path) -> None:
    _write_inputs(tmp_path, confidence=0.05)

    result = suggest_music_sync(tmp_path)

    assert result.aligned_boundary_count == 0
    assert result.boundaries[0].recommended_anchor == "none"
    assert result.boundaries[0].nearest_beat_seconds == 3.7


def test_music_sync_reports_unavailable_beat_grid(tmp_path: Path) -> None:
    _write_inputs(tmp_path, beats=[], downbeats=[])

    result = suggest_music_sync(tmp_path)

    assert result.aligned_boundary_count == 0
    assert result.boundaries[0].recommended_anchor == "none"
    assert any("No stable beat grid" in note for note in result.notes)


def test_music_sync_rejects_invalid_tolerance(tmp_path: Path) -> None:
    with pytest.raises(MusicSyncError, match="tolerance_seconds"):
        suggest_music_sync(tmp_path, tolerance_seconds=1.5)


def test_music_sync_rejects_artifacts_from_another_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _write_inputs(project)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(MusicSyncError, match="inside the project folder"):
        suggest_music_sync(project, timeline_path=other / "timeline.json")


def test_music_sync_report_serializable(tmp_path: Path) -> None:
    _write_inputs(tmp_path)

    result = suggest_music_sync(tmp_path)

    payload = json.loads(result.model_dump_json())
    assert payload["review_required"] is True
    assert payload["timeline_modified"] is False
