from pathlib import Path

from typer.testing import CliRunner

from video_edit_agent.cli import app

runner = CliRunner()


def test_clean_command_dry_run_preserves_frame(tmp_path: Path) -> None:
    frame = tmp_path / ".video-agent" / "frames" / "scene" / "frame-01.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"frame")

    result = runner.invoke(app, ["clean", str(tmp_path), "--max-age-days", "0", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Would remove 1 frame-cache file(s)" in result.output
    assert "Dry run only" in result.output
    assert frame.exists()


def test_clean_command_removes_old_cache_file(tmp_path: Path) -> None:
    import os
    import time

    frame = tmp_path / ".video-agent" / "highlight-frames" / "frame.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"frame")
    old = time.time() - 10 * 86400
    os.utime(frame, (old, old))

    result = runner.invoke(app, ["clean", str(tmp_path), "--max-age-days", "7"])

    assert result.exit_code == 0, result.output
    assert "Removed 1 frame-cache file(s)" in result.output
    assert not frame.exists()
