"""Tests for storyboard-driven create-video CLI orchestration."""

import click
import pytest
import typer.rich_utils
from typer.testing import CliRunner

from video_edit_agent.cli import app

runner = CliRunner()


@pytest.mark.parametrize("color", [False, True])
def test_create_video_help_exposes_storyboard_mode(tmp_path, monkeypatch, color):
    monkeypatch.setattr(typer.rich_utils, "FORCE_TERMINAL", color)
    result = runner.invoke(
        app,
        ["create-video", str(tmp_path), "--help"],
        color=True,
        env={"TERM": "xterm", "NO_COLOR": None},
    )

    assert result.exit_code == 0
    assert ("\x1b[" in result.output) is color
    help_text = click.unstyle(result.output)
    assert "--storyboard" in help_text
    assert "--highlights" in help_text
    assert "default" in " ".join(help_text.replace("│", " ").split()).lower()
    assert "storyboard" in help_text.lower()
