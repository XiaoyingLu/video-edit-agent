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
        "id: cinematic\nversion: 1\nmusic:\n  volume: 0.14\n", encoding="utf-8"
    )
    result = runner.invoke(app, ["template", "validate", "cinematic", "--project", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert str(directory / "cinematic.yaml") in result.output.replace("\n", "")


def test_template_show_reports_unknown_id():
    result = runner.invoke(app, ["template", "show", "not-a-template"])
    assert result.exit_code == 1
    assert "unknown template" in result.output


def test_template_apply_creates_review_only_proposal(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    import video_edit_agent.cli.templates as template_cli

    calls = {}
    overlay = SimpleNamespace(
        overlay_id="opening-1",
        sequence_role="opening",
        text="A journey",
        start_seconds=0.0,
        end_seconds=3.0,
        decision="pending",
    )
    document = SimpleNamespace(template_id="travel-montage", template_version=1, overlays=[overlay])
    destination = tmp_path / ".video-agent" / "title-overlays.json"

    def fake_suggest(folder, **kwargs):
        calls["folder"] = folder
        calls.update(kwargs)
        return document, destination

    monkeypatch.setattr(template_cli, "suggest_narrative_title_sequences", fake_suggest)
    result = runner.invoke(app, ["template", "apply", str(tmp_path), "travel-montage"])

    assert result.exit_code == 0, result.output
    assert calls["folder"] == tmp_path
    assert calls["template_id"] == "travel-montage"
    assert "pending" in result.output
    assert "nothing is approved automatically" in result.output


def test_template_apply_reports_missing_inputs(tmp_path: Path, monkeypatch):
    import video_edit_agent.cli.templates as template_cli

    def fail(*args, **kwargs):
        raise FileNotFoundError("Missing approved timeline")

    monkeypatch.setattr(template_cli, "suggest_narrative_title_sequences", fail)
    result = runner.invoke(app, ["template", "apply", str(tmp_path), "cinematic"])
    assert result.exit_code == 1
    assert "Missing approved timeline" in result.output
