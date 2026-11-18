"""Traceable, read-only summary of a complete rendered revision iteration."""

import html
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..artifact_tracking import register_artifact, validate_artifact


class RevisionCycleError(ValueError):
    """Raised when revision-cycle inputs cannot be assembled safely."""


class RevisionCycleArtifact(BaseModel):
    """One validated artifact included in the revision-cycle index."""

    model_config = ConfigDict(extra="forbid")

    role: str
    path: str
    summary: str


class RevisionCycleReport(BaseModel):
    """Index of the evidence and outputs for one human-reviewed revision cycle."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    project_path: str
    artifacts: list[RevisionCycleArtifact] = Field(default_factory=list)
    review_required: bool = True
    timeline_modified: bool = False
    source_media_modified: bool = False


def _project_artifact(root: Path, path: Path, role: str) -> Path:
    candidate = path.expanduser()
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RevisionCycleError(f"{role} must be inside the project folder") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {role}: {resolved}")
    if not validate_artifact(root, resolved):
        raise RevisionCycleError(f"{role} is not a current tracked artifact: {resolved}")
    return resolved


def _summary(role: str, path: Path) -> str:
    if path.suffix.lower() != ".json":
        return "Rendered video artifact"
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "JSON artifact"
    if not isinstance(payload, dict):
        return "JSON artifact"
    if role == "critique":
        return (
            f"{len(payload.get('findings', []))} findings; "
            f"{len(payload.get('semantic_reviews', []))} semantic reviews"
        )
    if role == "suggestions":
        return f"{len(payload.get('suggestions', []))} shot-level suggestions"
    if role == "preview":
        return (
            f"{payload.get('duration_seconds', 0):.2f}s · "
            f"{payload.get('width', '?')}×{payload.get('height', '?')}"
        )
    if role == "comparison":
        before = payload.get("before", {})
        after = payload.get("after", {})
        return (
            f"Duration {before.get('duration_seconds', '?')}s → "
            f"{after.get('duration_seconds', '?')}s; "
            f"{len(payload.get('frames', []))} paired frame samples"
        )
    if role == "timeline review":
        return (
            f"{len(payload.get('frames', []))} changed shots; "
            f"duration delta {payload.get('duration_delta_seconds', 0):+.2f}s"
        )
    return "Validated project artifact"


def build_revision_cycle_report(
    folder: Path,
    *,
    critique_path: Path,
    suggestions_path: Path,
    original_timeline_path: Path,
    revised_timeline_path: Path,
    preview_manifest_path: Path,
    comparison_path: Path,
    timeline_review_path: Path | None = None,
) -> RevisionCycleReport:
    """Validate and index the outputs of a revision iteration without changing them."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project folder does not exist: {root}")
    requested = [
        ("critique", critique_path),
        ("suggestions", suggestions_path),
        ("original timeline", original_timeline_path),
        ("revised timeline", revised_timeline_path),
        ("preview", preview_manifest_path),
        ("comparison", comparison_path),
    ]
    if timeline_review_path is not None:
        requested.append(("timeline review", timeline_review_path))
    artifacts = []
    for role, requested_path in requested:
        path = _project_artifact(root, requested_path, role)
        artifacts.append(
            RevisionCycleArtifact(
                role=role,
                path=path.relative_to(root).as_posix(),
                summary=_summary(role, path),
            )
        )
    return RevisionCycleReport(project_path=str(root), artifacts=artifacts)


def render_revision_cycle_html(report: RevisionCycleReport) -> str:
    """Render a self-contained HTML index with links to the original evidence."""
    rows = []
    for artifact in report.artifacts:
        path = html.escape(artifact.path, quote=True)
        rows.append(
            f'<tr><th>{html.escape(artifact.role)}</th>'
            f'<td><a href="{path}">{path}</a></td>'
            f'<td>{html.escape(artifact.summary)}</td></tr>'
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Revision Iteration Report</title>
<style>
body {{ max-width: 1000px; margin: 2rem auto; padding: 0 1rem; font: 16px system-ui,sans-serif; }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ text-align:left; padding:.8rem; border-bottom:1px solid #8886; }}
th {{ width:18%; }} a {{ overflow-wrap:anywhere; }}
.note {{ margin-top:2rem; padding:1rem; border:1px solid #8886; border-radius:8px; }}
</style></head><body>
<h1>Revision Iteration Report</h1>
<p>Project: {html.escape(report.project_path)}</p>
<p>This index connects the critique, revision proposal, timeline decisions, preview render,
and rendered comparison for one iteration. Open each artifact for its full evidence.</p>
<table><thead><tr><th>Stage</th><th>Artifact</th><th>Summary</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<div class="note">Human review remains required. This report does not approve timelines,
change source media, render videos, or determine which edit is aesthetically better.</div>
</body></html>"""


def save_revision_cycle_report(
    folder: Path,
    report: RevisionCycleReport,
    *,
    output_path: Path | None = None,
) -> tuple[Path, Path]:
    """Write a JSON and HTML index, refusing to overwrite prior reports."""
    root = folder.expanduser().resolve()
    base = (output_path or Path(".video-agent/revision-cycles/cycle-001")).expanduser()
    base = (root / base).resolve() if not base.is_absolute() else base.resolve()
    try:
        base.relative_to(root)
    except ValueError as exc:
        raise RevisionCycleError("Report output must remain inside the project folder") from exc
    json_file, html_file = base.with_suffix(".json"), base.with_suffix(".html")
    if json_file.exists() or html_file.exists():
        raise FileExistsError(f"Revision-cycle report already exists: {base}")
    json_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    html_file.write_text(render_revision_cycle_html(report), encoding="utf-8")
    dependencies = [root / item.path for item in report.artifacts]
    register_artifact(root, json_file, dependencies=dependencies)
    register_artifact(root, html_file, dependencies=[json_file, *dependencies])
    return json_file, html_file
