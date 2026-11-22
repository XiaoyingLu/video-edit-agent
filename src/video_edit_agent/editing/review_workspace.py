"""Generate a local landing page for the project's existing review artifacts."""

import html
import os
from pathlib import Path


def generate_review_workspace(folder: Path, *, output_path: Path | None = None) -> Path:
    """Create a project-local index of available visual reviews and next actions."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project folder does not exist: {root}")
    artifact_dir = root / ".video-agent"
    destination = (output_path or artifact_dir / "review-workspace.html").expanduser()
    destination = (
        (root / destination).resolve()
        if not destination.is_absolute()
        else destination.resolve()
    )
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("Workspace output must stay inside the project folder") from exc

    entries = [
        (
            "Storyboard",
            artifact_dir / "storyboard.html",
            "Inspect the narrative sequence and planned shots.",
        ),
        (
            "Timeline editor",
            _latest(artifact_dir / "timeline-visual-review", "*.html"),
            "Preview the sequence, reorder shots, adjust trims, and export decisions.",
        ),
        (
            "Soundtrack audition",
            _latest(artifact_dir, "soundtrack-review.html"),
            "Audition soundtrack candidates and review rights information.",
        ),
    ]
    cards = []
    for title, path, description in entries:
        available = path is not None and path.is_file()
        href = (
            html.escape(Path(os.path.relpath(path, destination.parent)).as_posix(), quote=True)
            if available
            else ""
        )
        action = (
            f'<a href="{href}">Open review</a>'
            if available
            else '<span class="missing">Not generated yet</span>'
        )
        cards.append(
            f'<article class="card"><h2>{html.escape(title)}</h2>'
            f'<p>{html.escape(description)}</p>{action}</article>'
        )

    data_files = [
        (
            "Rendered-draft critique",
            artifact_dir / "draft-critique.json",
            "critique-draft",
        ),
        (
            "Footage suggestions",
            artifact_dir / "footage-suggestions.json",
            "review-footage",
        ),
        (
            "Soundtrack selection",
            artifact_dir / "soundtrack-selection.json",
            "select-soundtrack",
        ),
    ]
    rows = []
    for label, path, command in data_files:
        status = "Available" if path.is_file() else "Not generated"
        rows.append(
            f"<tr><td>{html.escape(label)}</td><td>{status}</td>"
            f"<td><code>video-agent {html.escape(command)} ./PROJECT</code></td></tr>"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Video project review</title>
<style>
:root{color-scheme:light dark;font-family:system-ui,sans-serif}
body{max-width:1100px;margin:3rem auto;padding:0 1.25rem;line-height:1.5}
header{border-bottom:1px solid #8886;padding-bottom:1.2rem;margin-bottom:1.5rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1rem}
.card{border:1px solid #8886;border-radius:12px;padding:1.1rem}
.card h2{margin-top:0}.card p{min-height:3rem;color:GrayText}
a{font-weight:600}.missing{color:GrayText}
table{width:100%;border-collapse:collapse;margin-top:1rem}
th,td{text-align:left;padding:.7rem;border-bottom:1px solid #8886}
code{overflow-wrap:anywhere}
@media(max-width:650px){body{margin:1.5rem auto}table{font-size:.9rem}}
</style></head><body>
<header><h1>Project review</h1>
<p>A local index of this project's review tools and generated artifacts.</p>
<p><strong>Project:</strong> """
        + html.escape(root.name)
        + """</p></header>
<main><section><h2>Visual reviews</h2><div class="grid">"""
        + "".join(cards)
        + """</div></section><section><h2>Project artifacts</h2>
<p>Generate missing artifacts with the CLI commands below. Decisions and edits remain explicit;
this page does not approve clips or start a render.</p><table><thead><tr><th>Artifact</th>
<th>Status</th><th>Next action</th></tr></thead><tbody>"""
        + "".join(rows)
        + """</tbody></table></section></main></body></html>
""",
        encoding="utf-8",
    )
    return destination


def _latest(directory: Path, pattern: str) -> Path | None:
    if not directory.is_dir():
        return None
    matches = [path for path in directory.glob(pattern) if path.is_file()]
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None
