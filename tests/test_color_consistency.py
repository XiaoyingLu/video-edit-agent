import subprocess

import pytest

from video_edit_agent import color_consistency as color
from video_edit_agent.color_consistency import (
    ColorAnalysisError,
    ColorSample,
    _bounded_offset,
    _sample_signalstats,
)


def test_bounded_offsets_are_clamped_and_rounded():
    assert _bounded_offset(120, 80, 24) == 24
    assert _bounded_offset(80, 120, 24) == -24
    assert _bounded_offset(100.123, 99.0, 12) == 1.12


def test_signalstats_parser_extracts_luma_and_chroma(monkeypatch, tmp_path):
    monkeypatch.setattr(
        color,
        "run_ffmpeg",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            0,
            stdout=(
                "frame:0 pts:0\n"
                "lavfi.signalstats.YAVG=112.5\n"
                "lavfi.signalstats.UAVG=128\n"
                "lavfi.signalstats.VAVG=130.25\n"
            ),
            stderr="",
        ),
    )

    sample = _sample_signalstats("ffmpeg", tmp_path / "clip.mp4", 1.5)

    assert sample == ColorSample(
        timestamp_seconds=1.5, luma=112.5, chroma_u=128, chroma_v=130.25
    )


def test_signalstats_parser_rejects_missing_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(
        color,
        "run_ffmpeg",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="lavfi.signalstats.YAVG=100\n", stderr=""
        ),
    )

    with pytest.raises(ColorAnalysisError, match="UAVG"):
        _sample_signalstats("ffmpeg", tmp_path / "clip.mp4", 1.0)


def test_sample_count_must_be_bounded(tmp_path):
    with pytest.raises(ValueError, match="between 1 and 12"):
        color.analyze_project_color(tmp_path, sample_count=13)
