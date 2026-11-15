import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from travel_video_agent.cli import app
from travel_video_agent.models import ProjectAnalysis
from travel_video_agent.understanding.schemas import VideoUnderstanding

runner = CliRunner()


@pytest.mark.parametrize("output_option", [None, "--output", "-o"])
def test_analyze_command_writes_json(
    tmp_path: Path, monkeypatch, output_option: str | None
) -> None:
    from travel_video_agent import cli

    expected = {
        "project_path": str(tmp_path.resolve()),
        "video_count": 0,
        "total_duration_seconds": 0,
        "videos": [],
    }
    monkeypatch.setattr(cli, "analyze_folder", lambda folder: ProjectAnalysis(**expected))
    output = (
        tmp_path / "custom" / "analysis.json"
        if output_option
        else tmp_path / ".video-agent" / "analysis.json"
    )
    args = ["analyze", str(tmp_path)]
    if output_option:
        args.extend([output_option, str(output)])
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected


def test_understand_command_writes_json(tmp_path: Path, monkeypatch) -> None:
    from travel_video_agent import cli

    expected = VideoUnderstanding(
        project_path=str(tmp_path.resolve()),
        video_count=1,
        scene_count=0,
        scenes=[],
        model="qwen2.5vl:7b",
    )
    monkeypatch.setattr(cli, "understand_folder", lambda *args, **kwargs: expected)
    output = tmp_path / "custom" / "understanding.json"
    result = runner.invoke(app, ["understand", str(tmp_path), "--output", str(output)])
    assert result.exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected.model_dump(mode="json")


def test_cli_help_is_available() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "analyze" in result.output
    assert "understand" in result.output
