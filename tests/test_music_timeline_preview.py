from pathlib import Path

import pytest

from travel_video_agent.music.timeline_preview import (
    MusicTimelinePreview,
    MusicTimelinePreviewError,
    render_music_timeline_preview_html,
    save_music_timeline_preview,
)


def _report(root: Path) -> MusicTimelinePreview:
    soundtrack = root / "music track.mp3"
    soundtrack.write_bytes(b"audio")
    return MusicTimelinePreview(
        project_path=str(root.resolve()),
        timeline_path=".video-agent/optimized-timeline.json",
        timeline_sha256="abc123",
        music_analysis_path=".video-agent/music-analysis.json",
        music_path=str(soundtrack),
        duration_seconds=12,
        bpm=120,
        beat_confidence=0.8,
        beat_times_seconds=[0.5, 1.0, 1.5, 2.0],
        downbeat_times_seconds=[0.5, 2.5],
        sections=[
            {"start_seconds": 0, "end_seconds": 8, "energy": "low", "energy_rms": 0.1},
            {"start_seconds": 8, "end_seconds": 12, "energy": "high", "energy_rms": 0.4},
        ],
        clips=[
            {"clip_id": "opening", "start_seconds": 0, "end_seconds": 4, "duration_seconds": 4},
            {"clip_id": "coast", "start_seconds": 4, "end_seconds": 9, "duration_seconds": 5},
        ],
        boundaries=[
            {
                "previous_clip_id": "opening",
                "next_clip_id": "coast",
                "boundary_seconds": 4,
                "transition": "cut",
                "transition_duration_seconds": 0,
                "nearest_beat_seconds": 4.0,
                "beat_delta_seconds": 0.0,
                "nearest_downbeat_seconds": 4.5,
                "downbeat_delta_seconds": 0.5,
                "recommended_anchor": "beat",
                "recommended_boundary_seconds": 4.0,
                "adjustment_seconds": 0.0,
            }
        ],
    )


def test_html_shows_energy_beats_downbeats_and_shot_boundaries(tmp_path):
    report = _report(tmp_path)

    rendered = render_music_timeline_preview_html(report, audio_source="../music.mp3")

    assert "Music &amp; Timeline Sync Review" not in rendered
    assert "Music & Timeline Sync Review" in rendered
    assert "energy-low" in rendered
    assert "energy-high" in rendered
    assert "class=\"beat\"" in rendered
    assert "class=\"downbeat\"" in rendered
    assert "class=\"boundary\"" in rendered
    assert "class=\"recommendation\"" in rendered
    assert 'src="../music.mp3"' in rendered
    assert "not edit, approve, or render" in rendered


def test_html_handles_missing_tempo(tmp_path):
    report = _report(tmp_path).model_copy(update={"bpm": None})

    rendered = render_music_timeline_preview_html(report)

    assert "Tempo undetected" in rendered
    assert "Audio path unavailable" in rendered


def test_save_writes_relative_audio_path_and_refuses_overwrite(tmp_path, monkeypatch):
    report = _report(tmp_path)
    monkeypatch.setattr(
        "travel_video_agent.music.timeline_preview.register_artifact",
        lambda *_args, **_kwargs: None,
    )

    json_path, html_path = save_music_timeline_preview(tmp_path, report)

    assert json_path.is_file()
    assert html_path.is_file()
    assert 'src="../../music%20track.mp3"' in html_path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        save_music_timeline_preview(tmp_path, report)


def test_save_rejects_output_outside_project(tmp_path):
    report = _report(tmp_path)

    with pytest.raises(MusicTimelinePreviewError, match="inside the project"):
        save_music_timeline_preview(tmp_path, report, output_path=tmp_path.parent / "outside")
