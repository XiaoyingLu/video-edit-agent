"""Browser-based visual review for a proposed shot timeline."""

import base64
import hashlib
import html
import json
import shutil
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..artifact_tracking import register_artifact, validate_artifact
from ..rendering.critique import DraftCritiqueError, _run
from .timeline_schemas import ProjectDraftTimeline


class TimelineVisualReviewError(ValueError):
    """Raised when a visual timeline review cannot be generated safely."""


class TimelineVisualReviewClip(BaseModel):
    """Review context and representative source frame for one proposed shot."""

    model_config = ConfigDict(extra="forbid")

    clip_id: str
    sequence_index: int
    video_path: str
    source_start_seconds: float
    source_end_seconds: float
    duration_seconds: float
    scene_summary: str | None = None
    transcript_excerpt: str | None = None
    shot_scale: str
    visual_role: str
    mood: str
    temporal_summary: str
    continuity_note: str
    sequence_reason: str
    pacing_note: str
    music_energy: str
    music_pacing_note: str
    transition_in: str
    transition_duration_seconds: float
    decision: str
    frame_data_uri: str = Field(default="", exclude=True)


class TimelineVisualReviewReport(BaseModel):
    """Self-contained review data; frame bytes are embedded only in HTML."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    project_path: str
    timeline_path: str
    timeline_sha256: str
    proposed_duration_seconds: float
    target_duration_seconds: float
    timeline_approved: bool
    clips: list[TimelineVisualReviewClip] = Field(default_factory=list)
    review_required: bool = True
    timeline_modified: bool = False


class TimelineVisualReviewDecisions(BaseModel):
    """Decision export created by the browser review page."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    timeline_path: str
    timeline_sha256: str
    decisions: dict[str, str]


def _source_video(root: Path, value: str) -> Path:
    path = Path(value)
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise TimelineVisualReviewError(
            "Timeline source media must remain inside the project folder"
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing timeline source media: {resolved}")
    return resolved


def _frame(ffmpeg: str, video: Path, timestamp: float, destination: Path) -> str:
    try:
        result = _run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                "scale=720:-2:force_original_aspect_ratio=decrease",
                "-q:v",
                "4",
                "-y",
                str(destination),
            ],
            timeout=60,
        )
    except DraftCritiqueError as exc:
        raise TimelineVisualReviewError(str(exc)) from exc
    if result.returncode != 0 or not destination.is_file():
        raise TimelineVisualReviewError(
            f"Could not extract review frame from {video.name} at {timestamp:.2f}s: "
            f"{result.stderr.strip()}"
        )
    encoded = base64.b64encode(destination.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_timeline_visual_review(
    folder: Path, *, timeline_path: Path | None = None
) -> TimelineVisualReviewReport:
    """Extract one midpoint frame per clip without changing the timeline or source media."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project folder does not exist: {root}")
    timeline_file = (
        timeline_path or root / ".video-agent" / "optimized-timeline.json"
    ).expanduser()
    timeline_file = (
        (root / timeline_file).resolve()
        if not timeline_file.is_absolute()
        else timeline_file.resolve()
    )
    try:
        timeline_file.relative_to(root)
    except ValueError as exc:
        raise TimelineVisualReviewError("Timeline must be inside the project folder") from exc
    if not timeline_file.is_file():
        raise FileNotFoundError(f"Missing timeline: {timeline_file}")
    if not validate_artifact(root, timeline_file):
        raise TimelineVisualReviewError("Timeline is stale or untracked; regenerate it first")
    timeline = ProjectDraftTimeline.model_validate_json(timeline_file.read_text(encoding="utf-8"))
    if Path(timeline.project_path).resolve() != root:
        raise TimelineVisualReviewError("Timeline belongs to a different project folder")
    ffmpeg = shutil.which("ffmpeg")
    if timeline.clips and ffmpeg is None:
        raise TimelineVisualReviewError("ffmpeg must be installed and available on PATH")

    clips = []
    with tempfile.TemporaryDirectory(prefix="video-agent-timeline-review-") as temporary:
        directory = Path(temporary)
        for index, clip in enumerate(timeline.clips):
            video = _source_video(root, clip.video_path)
            midpoint = (clip.source_start_seconds + clip.source_end_seconds) / 2
            frame = _frame(ffmpeg, video, midpoint, directory / f"clip-{index:04}.jpg")
            clips.append(
                TimelineVisualReviewClip(
                    clip_id=clip.clip_id,
                    sequence_index=clip.sequence_index,
                    video_path=clip.video_path,
                    source_start_seconds=clip.source_start_seconds,
                    source_end_seconds=clip.source_end_seconds,
                    duration_seconds=clip.duration_seconds,
                    scene_summary=clip.scene_summary,
                    transcript_excerpt=clip.transcript_excerpt,
                    shot_scale=clip.shot_scale,
                    visual_role=clip.visual_role,
                    mood=clip.mood,
                    temporal_summary=clip.temporal_summary,
                    continuity_note=clip.continuity_note,
                    sequence_reason=clip.sequence_reason,
                    pacing_note=clip.pacing_note,
                    music_energy=clip.music_energy,
                    music_pacing_note=clip.music_pacing_note,
                    transition_in=clip.transition_in,
                    transition_duration_seconds=clip.transition_duration_seconds,
                    decision=clip.decision,
                    frame_data_uri=frame,
                )
            )
    return TimelineVisualReviewReport(
        project_path=str(root),
        timeline_path=timeline_file.relative_to(root).as_posix(),
        timeline_sha256=hashlib.sha256(timeline_file.read_bytes()).hexdigest(),
        proposed_duration_seconds=timeline.proposed_duration_seconds,
        target_duration_seconds=timeline.target_duration_seconds,
        timeline_approved=timeline.timeline_approved,
        clips=clips,
    )


def render_timeline_visual_review_html(report: TimelineVisualReviewReport) -> str:
    """Render an offline review page that exports decisions but never applies them."""
    payload = report.model_dump(mode="json")
    payload_json = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
    cards = []
    for clip in report.clips:
        image = clip.frame_data_uri
        details = [
            ("Scene", clip.scene_summary or "Not available"),
            ("Action", clip.temporal_summary or "Not available"),
            ("Editorial role", f"{clip.visual_role} · {clip.shot_scale} · {clip.mood}"),
            ("Sequence rationale", clip.sequence_reason),
            ("Continuity", clip.continuity_note),
            ("Pacing", clip.pacing_note),
            ("Music", f"{clip.music_energy} · {clip.music_pacing_note}"),
            (
                "Incoming transition",
                f"{clip.transition_in} · {clip.transition_duration_seconds:.2f}s",
            ),
            ("Transcript", clip.transcript_excerpt or "Not available"),
        ]
        detail_html = "".join(
            f"<p><strong>{html.escape(label)}:</strong> {html.escape(value)}</p>"
            for label, value in details
        )
        cards.append(
            f'<article class="clip" data-clip-id="{html.escape(clip.clip_id, quote=True)}">'
            f'<div class="image"><img src="{image}" alt="Representative frame for '
            f'{html.escape(clip.clip_id, quote=True)}"></div>'
            f'<div class="body"><div class="heading"><h2>{clip.sequence_index}. '
            f'{html.escape(clip.clip_id)}</h2><span class="duration">'
            f"{clip.duration_seconds:.2f}s</span></div>"
            f'<p class="source">{html.escape(clip.video_path)} · source '
            f"{clip.source_start_seconds:.2f}–{clip.source_end_seconds:.2f}s</p>"
            f'<div class="trim"><label>In (s) <input class="trim-start" type="number" '
            f'min="{clip.source_start_seconds:.3f}" max="{clip.source_end_seconds - 1:.3f}" '
            f'step="0.1" value="{clip.source_start_seconds:.3f}"></label>'
            f'<label>Out (s) <input class="trim-end" type="number" '
            f'min="{clip.source_start_seconds + 1:.3f}" max="{clip.source_end_seconds:.3f}" '
            f'step="0.1" value="{clip.source_end_seconds:.3f}"></label></div>'
            f"{detail_html}<fieldset><legend>Decision</legend>"
            f'<label><input type="radio" name="decision-{clip.sequence_index}" value="approved" '
            f"{'checked' if clip.decision == 'approved' else ''}> Approve</label>"
            f'<label><input type="radio" name="decision-{clip.sequence_index}" value="rejected" '
            f"{'checked' if clip.decision == 'rejected' else ''}> Reject</label>"
            f'<label><input type="radio" name="decision-{clip.sequence_index}" value="pending" '
            f"{'checked' if clip.decision == 'pending' else ''}> Skip / pending</label>"
            f"</fieldset></div></article>"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visual Timeline Review</title>
<style>
:root {{ color-scheme: light dark; font-family:system-ui,sans-serif; }}
body {{ max-width:1200px; margin:2rem auto; padding:0 1rem; }}
header {{ position:sticky; top:0; z-index:2; padding:1rem; background:Canvas;
          border-bottom:1px solid #8886; }}
.clip {{ display:grid; grid-template-columns:minmax(240px,40%) 1fr; gap:1rem;
         padding:1.25rem 0; border-bottom:1px solid #8886; }}
.clip.dragging {{ opacity:.45; }} .clip.drag-over {{ outline:2px dashed #888; }}
.image img {{ width:100%; height:auto; display:block; border-radius:8px; }}
.trim {{ display:flex; flex-wrap:wrap; gap:1rem; margin:.8rem 0; }}
.trim input {{ width:7rem; padding:.35rem; }} .clip {{ cursor:grab; }}
.heading {{ display:flex; justify-content:space-between; gap:1rem; align-items:baseline; }}
h2 {{ margin:.2rem 0; }} p {{ margin:.45rem 0; }}
.source {{ color:GrayText; overflow-wrap:anywhere; }}
fieldset {{ display:flex; flex-wrap:wrap; gap:1rem; border:1px solid #8886;
            border-radius:8px; margin-top:1rem; }}
button {{ padding:.65rem 1rem; cursor:pointer; }} .duration {{ font-variant-numeric:tabular-nums; }}
@media(max-width:700px) {{ .clip {{ grid-template-columns:1fr; }} header {{ position:static; }} }}
</style></head><body>
<header><h1>Visual Timeline Review</h1>
<p>{len(report.clips)} shots · {report.proposed_duration_seconds:.2f}s proposed ·
{report.target_duration_seconds:.2f}s target</p>
<p>Drag shots to reorder them and adjust source in/out points. Changes are exported as a
separate proposal; they are not applied until you import them with the CLI.</p>
<button id="export">Export visual edits (.json)</button>
<button id="export-decisions">Export review decisions (.json)</button>
<span id="status" aria-live="polite"></span></header>
<main>{"".join(cards) if cards else "<p>No clips are present in this timeline.</p>"}</main>
<script>
const report = {payload_json};
const main = document.querySelector("main");
let dragged = null;
for (const card of main.querySelectorAll(".clip")) {{
  card.draggable = true;
  card.addEventListener("dragstart", () => {{ dragged = card; card.classList.add("dragging"); }});
  card.addEventListener("dragend", () => {{
    card.classList.remove("dragging");
    for (const item of main.querySelectorAll(".clip")) item.classList.remove("drag-over");
  }});
  card.addEventListener("dragover", event => {{
    event.preventDefault();
    if (dragged && dragged !== card) card.classList.add("drag-over");
  }});
  card.addEventListener("dragleave", () => card.classList.remove("drag-over"));
  card.addEventListener("drop", event => {{
    event.preventDefault();
    if (!dragged || dragged === card) return;
    const bounds = card.getBoundingClientRect();
    main.insertBefore(dragged, event.clientY < bounds.top + bounds.height / 2 ? card : card.nextSibling);
    card.classList.remove("drag-over");
    Array.from(main.querySelectorAll(".clip")).forEach((item, index) => {{
      item.querySelector("h2").textContent = (index + 1) + ". " + item.dataset.clipId;
    }});
  }});
}}
function downloadJson(payload, filename) {{
  const blob = new Blob([JSON.stringify(payload, null, 2) + "\\n"], {{type:"application/json"}});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}}
document.getElementById("export").addEventListener("click", () => {{
  const clips = Array.from(main.querySelectorAll(".clip")).map(card => ({{
    clip_id: card.dataset.clipId,
    source_start_seconds: Number(card.querySelector(".trim-start").value),
    source_end_seconds: Number(card.querySelector(".trim-end").value)
  }}));
  downloadJson({{schema_version:"1.0", timeline_path:report.timeline_path,
    timeline_sha256:report.timeline_sha256, clips}}, "timeline-visual-edits.json");
  document.getElementById("status").textContent = " Visual edit proposal exported.";
}});
document.getElementById("export-decisions").addEventListener("click", () => {{
  const decisions = {{}};
  for (const clip of report.clips) {{
    const selected = document.querySelector('input[name="decision-' + clip.sequence_index + '"]:checked');
    decisions[clip.clip_id] = selected ? selected.value : "pending";
  }}
  downloadJson({{schema_version:"1.0", timeline_path:report.timeline_path,
    timeline_sha256:report.timeline_sha256, decisions}}, "timeline-review-decisions.json");
  document.getElementById("status").textContent = " Review decisions exported.";
}});
</script></body></html>"""


def save_timeline_visual_review(
    folder: Path,
    report: TimelineVisualReviewReport,
    *,
    output_path: Path | None = None,
) -> tuple[Path, Path]:
    """Save review metadata and HTML without overwriting earlier review artifacts."""
    root = folder.expanduser().resolve()
    base = (output_path or root / ".video-agent" / "timeline-visual-review" / "review").expanduser()
    base = (root / base).resolve() if not base.is_absolute() else base.resolve()
    try:
        base.relative_to(root)
    except ValueError as exc:
        raise TimelineVisualReviewError(
            "Review output must stay inside the project folder"
        ) from exc
    json_file, html_file = base.with_suffix(".json"), base.with_suffix(".html")
    if json_file.exists() or html_file.exists():
        raise FileExistsError(f"Refusing to overwrite visual timeline review: {base}")
    json_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    html_file.write_text(render_timeline_visual_review_html(report), encoding="utf-8")
    dependency = root / report.timeline_path
    register_artifact(root, json_file, dependencies=[dependency])
    register_artifact(root, html_file, dependencies=[dependency, json_file])
    return json_file, html_file


def load_timeline_visual_review_decisions(path: Path) -> TimelineVisualReviewDecisions:
    """Load and validate a decision export created by the browser review."""
    try:
        payload = TimelineVisualReviewDecisions.model_validate_json(
            path.expanduser().read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise TimelineVisualReviewError(f"Invalid visual review decisions: {exc}") from exc
    invalid = {
        clip_id: decision
        for clip_id, decision in payload.decisions.items()
        if decision not in {"approved", "rejected", "pending"}
    }
    if invalid:
        raise TimelineVisualReviewError(
            f"Unsupported decision for {sorted(invalid)[0]}; use approved, rejected, or pending"
        )
    return payload
