from __future__ import annotations

import subprocess

import pytest

from travel_video_agent.rendering import RenderError
from travel_video_agent.rendering.renderer import _validate_filter_graph


def test_filter_graph_preflight_uses_null_muxer_and_short_probe(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "travel_video_agent.rendering.renderer.subprocess.run", fake_run
    )
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

    monkeypatch.setattr(
        "travel_video_agent.rendering.renderer.subprocess.run", fake_run
    )
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
