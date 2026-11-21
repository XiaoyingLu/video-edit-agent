import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from video_edit_agent.cli import app
from video_edit_agent.editing.revision_cycle import (
    RevisionCycleError,
    build_revision_cycle_report,
    render_revision_cycle_html,
    save_revision_cycle_report,
)


def _artifacts(root: Path) -> dict[str, Path]:
    directory = root / ".video-agent"
    directory.mkdir(parents=True)
    payloads = {
        "critique": {"findings": [{"message": "Check pacing"}], "semantic_reviews": []},
        "suggestions": {"suggestions": [{"suggestion_id": "shot-1-pacing"}]},
        "original": {"timeline_approved": True},
        "revised": {"timeline_approved": True},
        "preview": {"duration_seconds": 12.5, "width": 1920, "height": 1080},
        "comparison": {
            "before": {"duration_seconds": 13},
            "after": {"duration_seconds": 12.5},
            "frames": [{}, {}],
        },
        "timeline_review": {"frames": [{"clip_id": "shot-1"}], "duration_delta_seconds": -0.5},
    }
    paths = {}
    for name, payload in payloads.items():
        path = directory / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    return paths


def _build(root, paths, monkeypatch, timeline_review=None, *, valid_artifact=True):
    monkeypatch.setattr(
        "video_edit_agent.editing.revision_cycle.validate_artifact", lambda *_: valid_artifact
    )
    return build_revision_cycle_report(
        root,
        critique_path=paths["critique"],
        suggestions_path=paths["suggestions"],
        original_timeline_path=paths["original"],
        revised_timeline_path=paths["revised"],
        preview_manifest_path=paths["preview"],
        comparison_path=paths["comparison"],
        timeline_review_path=timeline_review,
    )


def test_revision_cycle_report_indexes_all_stages(tmp_path, monkeypatch):
    paths = _artifacts(tmp_path)
    report = _build(tmp_path, paths, monkeypatch, paths["timeline_review"])

    assert [item.role for item in report.artifacts] == [
        "critique",
        "suggestions",
        "original timeline",
        "revised timeline",
        "preview",
        "comparison",
        "timeline review",
    ]
    assert "1 findings" in report.artifacts[0].summary
    assert "12.50s" in report.artifacts[4].summary
    assert "2 paired frame samples" in report.artifacts[5].summary
    assert report.review_required is True
    assert report.timeline_modified is False


def test_revision_cycle_rejects_artifact_outside_project(tmp_path, monkeypatch):
    paths = _artifacts(tmp_path)
    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    paths["critique"] = outside

    with pytest.raises(RevisionCycleError, match="inside the project"):
        _build(tmp_path, paths, monkeypatch)


def test_revision_cycle_rejects_stale_or_untracked_artifact(tmp_path, monkeypatch):
    paths = _artifacts(tmp_path)

    with pytest.raises(RevisionCycleError, match="not a current tracked artifact"):
        _build(tmp_path, paths, monkeypatch, valid_artifact=False)


def test_revision_cycle_html_escapes_paths_and_summaries(tmp_path):
    from video_edit_agent.editing.revision_cycle import (
        RevisionCycleArtifact,
        RevisionCycleReport,
    )

    report = RevisionCycleReport(
        project_path=str(tmp_path),
        artifacts=[
            RevisionCycleArtifact(
                role="critique",
                path=".video-agent/<script>.json",
                summary="<script>alert(1)</script>",
            )
        ],
    )
    rendered = render_revision_cycle_html(report)

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_revision_cycle_saves_json_and_html_with_dependencies(tmp_path, monkeypatch):
    paths = _artifacts(tmp_path)
    report = _build(tmp_path, paths, monkeypatch)
    registered = []
    monkeypatch.setattr(
        "video_edit_agent.editing.revision_cycle.register_artifact",
        lambda root, path, dependencies: registered.append((path, dependencies)),
    )

    json_file, html_file = save_revision_cycle_report(
        tmp_path, report, output_path=Path(".video-agent/revision-cycles/test-cycle")
    )

    assert json_file.is_file()
    assert html_file.is_file()
    assert len(registered) == 2
    assert len(registered[0][1]) == 6
    assert json_file.with_suffix(".html") == html_file
    with pytest.raises(FileExistsError):
        save_revision_cycle_report(
            tmp_path, report, output_path=Path(".video-agent/revision-cycles/test-cycle")
        )


def test_revision_cycle_cli_writes_report(tmp_path, monkeypatch):
    paths = _artifacts(tmp_path)
    monkeypatch.setattr(
        "video_edit_agent.editing.revision_cycle.validate_artifact", lambda *_: True
    )
    monkeypatch.setattr(
        "video_edit_agent.editing.revision_cycle.register_artifact",
        lambda *_args, **_kwargs: None,
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "report-revision-cycle",
            str(tmp_path),
            "--critique",
            str(paths["critique"]),
            "--suggestions",
            str(paths["suggestions"]),
            "--original-timeline",
            str(paths["original"]),
            "--revised-timeline",
            str(paths["revised"]),
            "--preview-manifest",
            str(paths["preview"]),
            "--comparison",
            str(paths["comparison"]),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Revision Iteration" in result.output
    assert (tmp_path / ".video-agent/revision-cycles/cycle-001.html").is_file()
