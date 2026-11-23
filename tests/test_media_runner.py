import subprocess

import pytest

from video_edit_agent.media_runner import MediaError, run_ffmpeg


def test_run_ffmpeg_returns_completed_process_and_uses_shared_options(monkeypatch):
    expected = subprocess.CompletedProcess(["ffmpeg", "-version"], 0, "version", "")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return expected

    monkeypatch.setattr("video_edit_agent.media_runner.subprocess.run", fake_run)

    result = run_ffmpeg(["ffmpeg", "-version"], timeout=12, label="FFmpeg version check")

    assert result is expected
    assert calls == [
        (
            ["ffmpeg", "-version"],
            {"capture_output": True, "text": True, "timeout": 12, "check": False},
        )
    ]


def test_run_ffmpeg_raises_common_error_for_timeout(monkeypatch):
    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr("video_edit_agent.media_runner.subprocess.run", fake_run)

    with pytest.raises(MediaError, match="frame extraction timed out after 3 seconds"):
        run_ffmpeg(["ffmpeg", "-i", "input.mp4"], timeout=3, label="frame extraction")


def test_run_ffmpeg_raises_common_error_with_stderr_for_nonzero_exit(monkeypatch):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, "", "invalid filter graph")

    monkeypatch.setattr("video_edit_agent.media_runner.subprocess.run", fake_run)

    with pytest.raises(MediaError, match="render failed with exit code 1: invalid filter graph"):
        run_ffmpeg(["ffmpeg", "-i", "input.mp4"], timeout=10, label="render")


def test_run_ffmpeg_decodes_binary_stderr_for_errors(monkeypatch):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 2, b"", b"decoder failed")

    monkeypatch.setattr("video_edit_agent.media_runner.subprocess.run", fake_run)

    with pytest.raises(MediaError, match="decoder failed"):
        run_ffmpeg(
            ["ffmpeg", "-i", "input.mp4"],
            timeout=10,
            label="decode",
            text=False,
        )


def test_run_ffmpeg_wraps_missing_executable(monkeypatch):
    def fake_run(command, **kwargs):
        raise FileNotFoundError("ffmpeg not installed")

    monkeypatch.setattr("video_edit_agent.media_runner.subprocess.run", fake_run)

    with pytest.raises(MediaError, match="could not start"):
        run_ffmpeg(["ffmpeg", "-i", "input.mp4"], timeout=10, label="decode")


def test_run_ffmpeg_preserves_binary_stdout(monkeypatch):
    expected = subprocess.CompletedProcess(["ffmpeg"], 0, b"\x00\x01", b"")

    monkeypatch.setattr(
        "video_edit_agent.media_runner.subprocess.run",
        lambda *args, **kwargs: expected,
    )

    result = run_ffmpeg(["ffmpeg"], timeout=10, label="raw audio decode", text=False)

    assert result.stdout == b"\x00\x01"
