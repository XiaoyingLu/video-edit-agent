import json
from pathlib import Path

from typer.testing import CliRunner

from travel_video_agent.cli import app

runner = CliRunner()


def test_analyze_command_writes_json(
    tmp_path: Path, monkeypatch
) -> None:
    from travel_video_agent import cli

    expected = {
        "project_path": str(tmp_path.resolve()),
        "video_count": 0,
        "total_duration_seconds": 0,
        "videos": [],
    }

    class FakeAnalysis:
        def model_dump(self, mode: str) -> dict:
            assert mode == "json"
            return expected

    monkeypatch.setattr(cli, "analyze_folder", lambda folder: FakeAnalysis())
    result = runner.invoke(app, ["analyze", str(tmp_path)])

    assert result.exit_code == 0
    output = tmp_path / ".video-agent" / "analysis.json"
    assert json.loads(output.read_text(encoding="utf-8")) == expected


def test_cli_help_is_available() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "analyze" in result.output
