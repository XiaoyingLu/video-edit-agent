from pathlib import Path
from types import SimpleNamespace

import pytest

from video_edit_agent.rendering import frame_audit


def test_sample_times_include_both_sides_of_boundaries():
    samples = frame_audit._sample_times(8.0, [3.0, 6.0])

    assert 3.0 in samples
    assert 6.0 in samples
    assert any(abs(value - (3.0 - 1 / 30)) < 1e-5 for value in samples)
    assert any(abs(value - (3.0 + 1 / 30)) < 1e-5 for value in samples)


def test_audit_decodes_frames_in_two_orders_and_reports_clipping_skip(monkeypatch, tmp_path):
    video = tmp_path / "render.mp4"
    video.write_bytes(b"rendered")
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if "-vf" in command:
            return SimpleNamespace(returncode=0, stdout="lavfi.signalstats.YAVG=20", stderr="")
        timestamp = float(command[command.index("-ss") + 1])
        return SimpleNamespace(returncode=0, stdout=f"frame-{timestamp}".encode(), stderr=b"")

    monkeypatch.setattr(frame_audit.subprocess, "run", fake_run)
    monkeypatch.setattr(
        frame_audit.shutil,
        "which",
        lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None,
    )

    report = frame_audit.audit_rendered_frames(
        "/usr/bin/ffmpeg",
        video,
        8.0,
        [3.0],
        width=1920,
        height=1080,
        has_titles=True,
    )

    assert report["deterministic"] is True
    assert report["sample_count"] >= 4
    assert report["boundary_count"] == 1
    assert "Tesseract OCR is not installed" in report["warnings"][0]
    assert len([command for command in calls if "-frames:v" in command]) == report["sample_count"] * 2


def test_audit_fails_when_random_access_frames_are_not_deterministic(monkeypatch, tmp_path):
    video = tmp_path / "render.mp4"
    video.write_bytes(b"rendered")
    count = 0

    def fake_run(command, **_kwargs):
        nonlocal count
        count += 1
        return SimpleNamespace(returncode=0, stdout=f"frame-{count}".encode(), stderr=b"")

    monkeypatch.setattr(frame_audit.subprocess, "run", fake_run)
    monkeypatch.setattr(frame_audit.shutil, "which", lambda _name: None)

    with pytest.raises(frame_audit.FrameAuditError, match="non-deterministic"):
        frame_audit.audit_rendered_frames(
            "/usr/bin/ffmpeg",
            video,
            8.0,
            [],
            width=1920,
            height=1080,
            has_titles=False,
        )


def test_validate_title_font_rejects_missing_explicit_path(tmp_path):
    missing = tmp_path / "missing-font.ttf"

    with pytest.raises(frame_audit.FrameAuditError, match="font file does not exist"):
        frame_audit.validate_title_font(str(missing), tmp_path / "bundled.ttf")


def test_validate_title_font_returns_bundled_font(tmp_path):
    bundled = tmp_path / "Lato-Regular.ttf"
    bundled.write_bytes(b"font")

    assert frame_audit.validate_title_font("Lato", bundled) == bundled.resolve()
