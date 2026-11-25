import subprocess

from video_edit_agent.models import ProjectAnalysis

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



def test_project_analysis_uses_one_project_wide_baseline(tmp_path, monkeypatch):
    metadata = tmp_path / ".video-agent"
    metadata.mkdir()
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    analysis_path = metadata / "analysis.json"
    analysis_path.write_text(
        ProjectAnalysis(
            project_path=str(tmp_path.resolve()),
            video_count=2,
            total_duration_seconds=4,
            videos=[
                {
                    "path": "first.mp4",
                    "duration_seconds": 2,
                    "has_video": True,
                    "has_audio": False,
                },
                {
                    "path": "second.mp4",
                    "duration_seconds": 2,
                    "has_video": True,
                    "has_audio": False,
                },
            ],
        ).model_dump_json(),
        encoding="utf-8",
    )
    monkeypatch.setattr(color, "validate_artifact", lambda *args, **kwargs: False)
    monkeypatch.setattr(color.shutil, "which", lambda _: "ffmpeg")

    def sample(_ffmpeg, source, timestamp):
        if source.name == "first.mp4":
            return ColorSample(timestamp_seconds=timestamp, luma=80, chroma_u=128, chroma_v=128)
        return ColorSample(timestamp_seconds=timestamp, luma=120, chroma_u=140, chroma_v=136)

    monkeypatch.setattr(color, "_sample_signalstats", sample)

    report = color.analyze_project_color(tmp_path, sample_count=1)

    assert [video.baseline_luma for video in report.videos] == [100, 100]
    assert report.videos[0].suggestions[0].luma_offset == 20
    assert report.videos[1].suggestions[0].luma_offset == -20
    assert report.approval_required is True
    assert report.corrections_applied is False
    assert report.rendering_performed is False
