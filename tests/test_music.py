from pathlib import Path

import numpy as np
import pytest

from travel_video_agent.music.pipeline import (
    _beat_grid,
    _energy_sections,
    _estimate_tempo,
    _onset_envelope,
    analyze_music,
)
from travel_video_agent.music.schemas import MusicAnalysis


def test_estimates_regular_120_bpm_onset_train():
    times = np.arange(0, 30, 0.01)
    onset = np.zeros_like(times)
    onset[np.arange(0, len(times), 50)] = 1
    bpm, confidence, interval = _estimate_tempo(times, onset)
    assert bpm == pytest.approx(120, abs=2)
    assert confidence > 0
    assert interval == pytest.approx(0.5, abs=0.02)


def test_beat_grid_returns_ordered_times():
    times = np.arange(0, 8, 0.01)
    onset = np.zeros_like(times)
    onset[np.arange(10, len(times), 50)] = 1
    beats = _beat_grid(times, onset, 0.5, 8)
    assert len(beats) >= 10
    assert beats == sorted(beats)
    assert all(0 <= beat < 8 for beat in beats)


def test_energy_sections_cover_track_and_use_relative_levels():
    samples = np.concatenate(
        [np.full(8 * 11025, 0.02), np.full(8 * 11025, 0.5), np.full(8 * 11025, 0.1)]
    ).astype(np.float32)
    sections = _energy_sections(samples, 24)
    assert len(sections) == 3
    assert sections[0].energy == "low"
    assert sections[1].energy == "high"
    assert sections[-1].end_seconds == 24


def test_onset_envelope_handles_synthetic_audio():
    rate = 11025
    t = np.arange(rate * 5) / rate
    samples = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    times, onset = _onset_envelope(samples)
    assert len(times) == len(onset)
    assert np.isfinite(onset).all()


def test_music_analysis_schema_rejects_invalid_beat_positions():
    payload = {
        "project_path": "/tmp/project",
        "music_path": "/tmp/song.wav",
        "duration_seconds": 10,
        "sample_rate": 11025,
        "beat_confidence": 0.5,
        "beat_times_seconds": [-1],
    }
    with pytest.raises(ValueError):
        MusicAnalysis.model_validate(payload)


def test_analyze_music_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        analyze_music(tmp_path, tmp_path / "missing.mp3")
