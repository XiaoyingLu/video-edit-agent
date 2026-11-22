"""Tests for storyboard-driven create-video CLI orchestration."""

from typer.testing import CliRunner

from video_edit_agent.cli import app


runner = CliRunner()


def test_create_video_help_exposes_storyboard_mode(tmp_path):
    result = runner.invoke(app, ["create-video", str(tmp_path), "--help"])

    assert result.exit_code == 0
    assert "--storyboard" in result.output
    assert "semantic footage matching" in result.output
