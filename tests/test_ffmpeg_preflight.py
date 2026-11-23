from __future__ import annotations

import subprocess

import pytest

from video_edit_agent.rendering import RenderError
from video_edit_agent.rendering.shot_renderer import _validate_xfade_transitions
from video_edit_agent.rendering.renderer import (
    _validate_filter_graph,
    _validate_source_stream_metadata,
)


def test_filter_graph_preflight_uses_null_muxer_and_short_probe(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("video_edit_agent.media_runner.subprocess.run", fake_run)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        "input.mp4",
        "-filter_complex",
        "[0:v]null[v]",
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "output.mp4",
    ]

    _validate_filter_graph(command, timeout=30)

    assert captured["command"][-5:] == ["-t", "0.5", "-f", "null", "-"]
    assert "-filter_complex" in captured["command"]
    assert "libx264" not in captured["command"]
    assert captured["kwargs"]["timeout"] == 30


def test_filter_graph_preflight_surfaces_ffmpeg_diagnostics(monkeypatch):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="Invalid filter graph")

    monkeypatch.setattr("video_edit_agent.media_runner.subprocess.run", fake_run)
    command = [
        "ffmpeg",
        "-filter_complex",
        "[0:v]broken",
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "output.mp4",
    ]

    with pytest.raises(RenderError, match="Invalid filter graph"):
        _validate_filter_graph(command)


def test_filter_graph_preflight_rejects_missing_encoder_marker():
    with pytest.raises(RenderError, match="encoder options"):
        _validate_filter_graph(["ffmpeg", "-filter_complex", "null"])


def test_xfade_preflight_accepts_supported_transition_names(monkeypatch, tmp_path):
    help_text = """
    transition <int> (default fade)
       fade 0
       dissolve 1
       fadeblack 2
       wipeleft 3
       slideleft 4
       circleopen 5
    """

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=help_text, stderr="")

    monkeypatch.setattr("video_edit_agent.media_runner.subprocess.run", fake_run)
    clip_inputs = [
        (tmp_path / "clip.mp4", 0, 2, True, "fadeblack", 0.4),
        (tmp_path / "clip.mp4", 2, 4, True, "wipeleft", 0.4),
        (tmp_path / "clip.mp4", 4, 6, True, "slideleft", 0.4),
        (tmp_path / "clip.mp4", 6, 8, True, "circleopen", 0.4),
    ]

    _validate_xfade_transitions("ffmpeg", clip_inputs)


def test_xfade_preflight_rejects_transition_missing_from_installed_build(monkeypatch, tmp_path):
    help_text = """
    transition <int> (default fade)
       fade 0
       dissolve 1
       wipeleft 3
    """

    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout=help_text, stderr="")

    monkeypatch.setattr("video_edit_agent.media_runner.subprocess.run", fake_run)
    clip_inputs = [(tmp_path / "clip.mp4", 0, 2, True, "circleopen", 0.4)]

    with pytest.raises(RenderError, match="does not support.*circleopen"):
        _validate_xfade_transitions("ffmpeg", clip_inputs)


def test_xfade_preflight_skips_cut_only_timelines(monkeypatch, tmp_path):
    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("xfade capability check should not run for cut-only timelines")

    monkeypatch.setattr("video_edit_agent.media_runner.subprocess.run", unexpected_run)
    _validate_xfade_transitions(
        "ffmpeg", [(tmp_path / "clip.mp4", 0, 2, True, "cut", 0)]
    )


def test_source_metadata_validation_checks_dimensions_and_audio_fields(tmp_path):
    payload = {
        "streams": [
            {"codec_type": "video", "width": 1920, "height": 1080},
            {"codec_type": "audio", "sample_rate": "48000", "channels": 2},
        ]
    }

    _validate_source_stream_metadata(payload, tmp_path / "clip.mp4")

    payload["streams"][1]["sample_rate"] = "0"
    with pytest.raises(RenderError, match="sample rate"):
        _validate_source_stream_metadata(payload, tmp_path / "clip.mp4")


def test_source_metadata_validation_rejects_invalid_video_dimensions(tmp_path):
    payload = {"streams": [{"codec_type": "video", "width": 0, "height": 1080}]}

    with pytest.raises(RenderError, match="dimensions"):
        _validate_source_stream_metadata(payload, tmp_path / "clip.mp4")
