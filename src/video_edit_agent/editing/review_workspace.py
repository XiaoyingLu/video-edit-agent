"""Generate a unified local review workspace for a project's existing review artifacts."""

import html
import os
from pathlib import Path


def generate_review_workspace(folder: Path, *, output_path: Path | None = None) -> Path:
    """Create one tabbed review surface around the project's existing local review pages."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project folder does not exist: {root}")
    artifact_dir = root / ".video-agent"
    destination = (output_path or artifact_dir / "review-workspace.html").expanduser()
    destination = (
        (root / destination).resolve() if not destination.is_absolute() else destination.resolve()
    )
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("Workspace output must stay inside the project folder") from exc

    entries = [
        (
            "storyboard",
            "Storyboard",
            artifact_dir / "storyboard.html",
            "Review the narrative sequence and planned shots.",
        ),
        (
            "timeline",
            "Timeline editor",
            _latest(artifact_dir / "timeline-visual-review", "*.html"),
            "Preview the sequence, reorder shots, adjust trims, and export a revision.",
        ),
        (
            "music",
            "Music sync",
            _latest(artifact_dir / "music-timeline-preview", "*.html"),
            "Listen to the soundtrack while inspecting beats, energy, and cut alignment.",
        ),
        (
            "soundtrack",
            "Soundtrack",
            artifact_dir / "soundtrack-review.html",
            "Audition soundtrack candidates and review their usage information.",
        ),
    ]

    tabs = []
    panels = []
    for index, (key, title, path, description) in enumerate(entries):
        available = path is not None and path.is_file()
        relative = (
            Path(os.path.relpath(path, destination.parent)).as_posix() if available else None
        )
        escaped_title = html.escape(title)
        if available:
            href = html.escape(relative or "", quote=True)
            tabs.append(
                f'<button class="tab" type="button" role="tab" id="tab-{key}" '
                f'aria-controls="panel-{key}" aria-selected="{"true" if index == 0 else "false"}" '
                f'tabindex="{"0" if index == 0 else "-1"}" data-panel="{key}">{escaped_title}</button>'
            )
            panels.append(
                f'<section class="panel" role="tabpanel" id="panel-{key}" '
                f'aria-labelledby="tab-{key}" {" " if index == 0 else "hidden"}>'
                f'<div class="panel-heading"><div><h2>{escaped_title}</h2>'
                f'<p>{html.escape(description)}</p></div>'
                f'<a href="{href}" target="_blank" rel="noopener">Open separately</a></div>'
                f'<iframe title="{escaped_title}" src="{href}" loading="{"eager" if index == 0 else "lazy"}" '
                f'sandbox="allow-scripts allow-same-origin allow-downloads"></iframe></section>'
            )
        else:
            tabs.append(
                f'<button class="tab" type="button" role="tab" id="tab-{key}" '
                f'aria-controls="panel-{key}" aria-selected="false" tabindex="-1" '
                f'data-panel="{key}" disabled>{escaped_title} · unavailable</button>'
            )
            panels.append(
                f'<section class="panel" role="tabpanel" id="panel-{key}" '
                f'aria-labelledby="tab-{key}" hidden><div class="empty">'
                f'<h2>{escaped_title}</h2><p>Not generated yet.</p>'
                f'<p>{html.escape(description)}</p></div></section>'
            )

    data_files = [
        ("Rendered-draft critique", artifact_dir / "draft-critique.json", "critique-draft"),
        ("Footage suggestions", artifact_dir / "footage-suggestions.json", "review-footage"),
        ("Soundtrack selection", artifact_dir / "soundtrack-selection.json", "select-soundtrack"),
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
body{margin:0;line-height:1.5}
header,.artifacts{max-width:1400px;margin:0 auto;padding:1.2rem 1.5rem}
header{border-bottom:1px solid #8886}
h1{margin:.2rem 0}.muted,.panel-heading p{color:GrayText}
.tabs{display:flex;gap:.4rem;flex-wrap:wrap;margin-top:1rem}
.tab{padding:.65rem .9rem;border:1px solid #8888;border-radius:8px;background:transparent;
     color:inherit;cursor:pointer;font:inherit}
.tab[aria-selected=true]{background:Highlight;color:HighlightText;border-color:Highlight}
.tab:disabled{opacity:.45;cursor:not-allowed}
.panel{max-width:1500px;margin:0 auto;padding:1rem 1.5rem}
.panel[hidden]{display:none}
.panel-heading{display:flex;align-items:center;justify-content:space-between;gap:1rem}
.panel-heading h2{margin:.2rem 0}.panel-heading p{margin:.2rem 0 1rem}
.panel-heading a{white-space:nowrap}
iframe{display:block;width:100%;height:min(78vh,1050px);min-height:620px;
       border:1px solid #8886;border-radius:10px;background:Canvas}
.empty{padding:3rem;border:1px dashed #8888;border-radius:12px}
.artifacts{border-top:1px solid #8886;margin-top:1rem}
table{width:100%;border-collapse:collapse;margin-top:1rem}
th,td{text-align:left;padding:.7rem;border-bottom:1px solid #8886}
code{overflow-wrap:anywhere}
@media(max-width:700px){header,.artifacts{padding:1rem}
.panel{padding:.75rem}iframe{min-height:520px;height:75vh}
.panel-heading{align-items:flex-start;flex-direction:column}}
</style></head><body>
<header><h1>Project review</h1>
<p class="muted">Review your storyboard, timeline, and soundtrack in one local workspace.</p>
<p><strong>Project:</strong> """
        + html.escape(root.name)
        + """</p><nav class="tabs" role="tablist" aria-label="Review sections">"""
        + "".join(tabs)
        + """</nav></header><main>"""
        + "".join(panels)
        + """</main><section class="artifacts"><h2>Project artifacts</h2>
<p class="muted">Some analysis and critique actions still use the CLI. Browser edits remain exports
until you explicitly apply and approve a revision; this workspace never starts a render.</p>
<table><thead><tr><th>Artifact</th><th>Status</th><th>Next action</th></tr></thead>
<tbody>"""
        + "".join(rows)
        + """</tbody></table></section>
<script>
const tabs = Array.from(document.querySelectorAll('[role="tab"]:not(:disabled)'));
function activateTab(selected, moveFocus) {
  for (const tab of tabs) {
    const active = tab === selected;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    document.getElementById(tab.getAttribute("aria-controls")).hidden = !active;
  }
  if (moveFocus) selected.focus();
}
for (const tab of tabs) {
  tab.addEventListener("click", () => activateTab(tab, false));
  tab.addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const index = tabs.indexOf(tab);
    const next = event.key === "Home" ? 0 :
      event.key === "End" ? tabs.length - 1 :
      (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
    activateTab(tabs[next], true);
  });
}
</script></body></html>
""",
        encoding="utf-8",
    )
    return destination


def _latest(directory: Path, pattern: str) -> Path | None:
    if not directory.is_dir():
        return None
    matches = [path for path in directory.glob(pattern) if path.is_file()]
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None
