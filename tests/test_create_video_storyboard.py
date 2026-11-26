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


def test_storyboard_playback_preview_opens_video_and_requires_final_confirmation(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from video_edit_agent.cli import planning_story

    preview = tmp_path / ".video-agent" / "revision-previews" / "timeline.mp4"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"preview")
    manifest = preview.with_suffix(".preview.json")
    calls = []
    monkeypatch.setattr(
        planning_story,
        "render_timeline_revision_preview",
        lambda *args, **kwargs: (
            SimpleNamespace(
                output_path=preview,
                width=1280,
                height=720,
                actual_duration_seconds=12.5,
            ),
            manifest,
        ),
    )
    monkeypatch.setattr(planning_story.webbrowser, "open", lambda url: calls.append(url))
    monkeypatch.setattr(planning_story.console, "input", lambda prompt: "yes")

    planning_story._review_storyboard_playback_preview(tmp_path, tmp_path / "timeline.json")

    assert calls == [preview.as_uri()]


def test_storyboard_playback_preview_preserves_timeline_when_user_declines(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from video_edit_agent.cli import planning_story

    preview = tmp_path / "preview.mp4"
    preview.write_bytes(b"preview")
    timeline = tmp_path / "timeline.json"
    timeline.write_bytes(b'{"approved": true}')
    monkeypatch.setattr(
        planning_story,
        "render_timeline_revision_preview",
        lambda *args, **kwargs: (
            SimpleNamespace(
                output_path=preview,
                width=1280,
                height=720,
                actual_duration_seconds=12.5,
            ),
            tmp_path / "preview.json",
        ),
    )
    monkeypatch.setattr(planning_story.webbrowser, "open", lambda url: None)
    monkeypatch.setattr(planning_story.console, "input", lambda prompt: "n")

    with pytest.raises(typer.Exit) as exc_info:
        planning_story._review_storyboard_playback_preview(tmp_path, timeline)

    assert exc_info.value.exit_code == 2
    assert timeline.read_bytes() == b'{"approved": true}'
    assert preview.read_bytes() == b"preview"
