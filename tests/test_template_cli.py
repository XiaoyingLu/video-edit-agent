from pathlib import Path

from typer.testing import CliRunner

from video_edit_agent.cli import app

runner = CliRunner()


def test_template_list_includes_builtins():
    result = runner.invoke(app, ["template", "list"])
    assert result.exit_code == 0, result.output
    for template_id in ("cinematic", "vlog", "travel-montage", "social"):
        assert template_id in result.output


def test_template_show_displays_resolved_template():
    result = runner.invoke(app, ["template", "show", "travel-montage"])
    assert result.exit_code == 0, result.output
    assert '"id": "travel-montage"' in result.output
    assert '"volume": 0.25' in result.output


def test_template_validate_accepts_builtin():
    result = runner.invoke(app, ["template", "validate", "cinematic"])
    assert result.exit_code == 0, result.output
    assert "Valid template" in result.output


def test_template_validate_uses_project_override(tmp_path: Path):
    directory = tmp_path / ".video-agent" / "templates"
    directory.mkdir(parents=True)
    (directory / "cinematic.yaml").write_text(
        "id: cinematic\\nversion: 1\\nmusic:\\n  volume: 0.14\\n", encoding="utf-8"
    )
    result = runner.invoke(app, ["template", "validate", "cinematic", "--project", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert str(directory / "cinematic.yaml") in result.output


def test_template_show_reports_unknown_id():
    result = runner.invoke(app, ["template", "show", "not-a-template"])
    assert result.exit_code == 1
    assert "unknown template" in result.output
