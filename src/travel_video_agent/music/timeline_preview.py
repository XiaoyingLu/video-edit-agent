"""Read-only browser preview of soundtrack energy, beat grid, and shot timing."""

import html
import json
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..artifact_tracking import register_artifact, validate_artifact
from ..editing.timeline_schemas import ProjectDraftTimeline
from .schemas import MusicAnalysis
from .sync import MusicSyncSuggestions


class MusicTimelinePreviewError(ValueError):
    """Raised when music/timeline preview inputs are unsafe or inconsistent."""


class MusicTimelinePreview(BaseModel):
    """Version-bound data for comparing soundtrack anchors with shot boundaries."""

    model_config = ConfigDict(extra="forbid")

    project_path: str
    timeline_path: str
    timeline_sha256: str
    music_analysis_path: str
    music_path: str
    duration_seconds: float = Field(gt=0)
    bpm: float | None = None
    beat_confidence: float = Field(ge=0, le=1)
    beat_times_seconds: list[float] = Field(default_factory=list)
    downbeat_times_seconds: list[float] = Field(default_factory=list)
    sections: list[dict] = Field(default_factory=list)
    clips: list[dict] = Field(default_factory=list)
    boundaries: list[dict] = Field(default_factory=list)
    review_only: bool = True
    timeline_modified: bool = False


def build_music_timeline_preview(
    folder: Path,
    *,
    timeline_path: Path | None = None,
    music_analysis_path: Path | None = None,
    suggestions_path: Path | None = None,
) -> MusicTimelinePreview:
    """Validate current project artifacts and combine their timing evidence."""
    import hashlib

    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project folder does not exist: {root}")
    timeline_file = (
        timeline_path or root / ".video-agent" / "optimized-timeline.json"
    ).expanduser()
    analysis_file = (
        music_analysis_path or root / ".video-agent" / "music-analysis.json"
    ).expanduser()
    suggestions_file = (
        suggestions_path or root / ".video-agent" / "music-sync-suggestions.json"
    ).expanduser()
    paths = []
    for candidate, label in (
        (timeline_file, "timeline"),
        (analysis_file, "music analysis"),
        (suggestions_file, "music sync suggestions"),
    ):
        resolved = (
            (root / candidate).resolve()
            if not candidate.is_absolute()
            else candidate.resolve()
        )
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise MusicTimelinePreviewError(f"{label} artifact must be inside the project") from exc
        if not resolved.is_file():
            raise FileNotFoundError(f"Missing {label} artifact: {resolved}")
        if not validate_artifact(root, resolved):
            raise MusicTimelinePreviewError(f"{label} artifact is stale or untracked")
        paths.append(resolved)
    timeline_file, analysis_file, suggestions_file = paths
    timeline = ProjectDraftTimeline.model_validate_json(timeline_file.read_text(encoding="utf-8"))
    music = MusicAnalysis.model_validate_json(analysis_file.read_text(encoding="utf-8"))
    suggestions = MusicSyncSuggestions.model_validate_json(
        suggestions_file.read_text(encoding="utf-8")
    )
    if Path(timeline.project_path).resolve() != root or Path(music.project_path).resolve() != root:
        raise MusicTimelinePreviewError("Input artifacts belong to a different project")
    if Path(suggestions.project_path).resolve() != root:
        raise MusicTimelinePreviewError("Music sync suggestions belong to a different project")
    if Path(suggestions.timeline_path).as_posix() != timeline_file.relative_to(root).as_posix():
        raise MusicTimelinePreviewError("Music sync suggestions reference a different timeline")
    if (
        Path(suggestions.music_analysis_path).as_posix()
        != analysis_file.relative_to(root).as_posix()
    ):
        raise MusicTimelinePreviewError("Music sync suggestions reference different music analysis")
    soundtrack = Path(music.music_path).expanduser().resolve()
    if not soundtrack.is_file() or soundtrack != Path(suggestions.music_path).resolve():
        raise MusicTimelinePreviewError(
            "Analyzed soundtrack is missing or does not match suggestions"
        )

    elapsed = 0.0
    clips = []
    for index, clip in enumerate(timeline.clips):
        clips.append({
            "clip_id": clip.clip_id,
            "start_seconds": round(elapsed, 3),
            "end_seconds": round(elapsed + clip.duration_seconds, 3),
            "duration_seconds": clip.duration_seconds,
        })
        elapsed += clip.duration_seconds - (
            clip.transition_duration_seconds if index else 0.0
        )
    boundaries = [item.model_dump(mode="json") for item in suggestions.boundaries]
    return MusicTimelinePreview(
        project_path=str(root),
        timeline_path=timeline_file.relative_to(root).as_posix(),
        timeline_sha256=hashlib.sha256(timeline_file.read_bytes()).hexdigest(),
        music_analysis_path=analysis_file.relative_to(root).as_posix(),
        music_path=str(soundtrack),
        duration_seconds=music.duration_seconds,
        bpm=music.bpm,
        beat_confidence=music.beat_confidence,
        beat_times_seconds=music.beat_times_seconds,
        downbeat_times_seconds=music.downbeat_times_seconds,
        sections=[section.model_dump(mode="json") for section in music.sections],
        clips=clips,
        boundaries=boundaries,
    )


def render_music_timeline_preview_html(
    report: MusicTimelinePreview, *, audio_source: str = ""
) -> str:
    """Render a single time-axis view with energy, beats, downbeats, and shot boundaries."""
    width = 1100
    left = 150
    right = 30
    plot_width = width - left - right
    duration = max(
        report.duration_seconds,
        max((clip["end_seconds"] for clip in report.clips), default=0.0),
        0.001,
    )

    def x_at(seconds: float) -> float:
        return left + max(0.0, min(duration, seconds)) / duration * plot_width

    energy_rects = []
    for section in report.sections:
        x1, x2 = x_at(section["start_seconds"]), x_at(section["end_seconds"])
        energy_rects.append(
            f'<rect x="{x1:.1f}" y="42" width="{max(1, x2-x1):.1f}" height="34" '
            f'class="energy energy-{section["energy"]}"><title>{section["energy"]} energy '
            f'{section["start_seconds"]:.1f}–{section["end_seconds"]:.1f}s</title></rect>'
        )
    beats = "".join(
        f'<line x1="{x_at(value):.1f}" x2="{x_at(value):.1f}" y1="88" y2="170" '
        f'class="beat"><title>Beat {value:.3f}s</title></line>'
        for value in report.beat_times_seconds
    )
    downbeats = "".join(
        f'<line x1="{x_at(value):.1f}" x2="{x_at(value):.1f}" y1="82" y2="176" '
        f'class="downbeat"><title>Downbeat {value:.3f}s</title></line>'
        for value in report.downbeat_times_seconds
    )
    clip_rows = []
    for index, clip in enumerate(report.clips):
        y = 205 + index * 38
        x1, x2 = x_at(clip["start_seconds"]), x_at(clip["end_seconds"])
        clip_rows.append(
            f'<text x="12" y="{y+17}" class="clip-label">{html.escape(clip["clip_id"])}</text>'
            f'<rect x="{x1:.1f}" y="{y}" width="{max(2,x2-x1):.1f}" height="23" '
            f'class="clip-bar"><title>{html.escape(clip["clip_id"])} '
            f'{clip["start_seconds"]:.2f}–{clip["end_seconds"]:.2f}s</title></rect>'
        )
    boundary_rows = []
    for item in report.boundaries:
        x = x_at(item["boundary_seconds"])
        recommended = item.get("recommended_boundary_seconds")
        marker = (
            f'<line x1="{x_at(recommended):.1f}" x2="{x_at(recommended):.1f}" '
            f'y1="185" y2="{205 + len(report.clips)*38}" class="recommendation">'
            f'<title>Suggested {item["recommended_anchor"]} at {recommended:.3f}s</title></line>'
            if recommended is not None
            else ""
        )
        boundary_rows.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="180" '
            f'y2="{205 + len(report.clips)*38}" class="boundary">'
            f'<title>{html.escape(item["previous_clip_id"])} → '
            f'{html.escape(item["next_clip_id"])} at {item["boundary_seconds"]:.3f}s</title></line>'
            + marker
        )
    ticks = "".join(
        f'<text x="{x_at(value):.1f}" y="25" text-anchor="middle" class="tick">'
        f'{value:.0f}s</text>'
        for value in range(0, int(duration) + 1, max(1, int(duration / 10)))
    )
    svg_height = 220 + len(report.clips) * 38
    payload = json.dumps(report.model_dump(mode="json"), ensure_ascii=True).replace("<", "\\u003c")
    audio_html = (
        f'<audio controls preload="metadata" src="{html.escape(audio_source, quote=True)}"></audio>'
        if audio_source
        else "<p>Audio path unavailable. Regenerate the preview from the project folder.</p>"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Music & Timeline Sync Review</title>
<style>
:root {{ color-scheme:light dark; font-family:system-ui,sans-serif; }}
body {{ max-width:1200px; margin:2rem auto; padding:0 1rem; }}
.track {{ overflow-x:auto; border:1px solid #8886; border-radius:10px; padding:1rem; }}
svg {{ min-width:850px; width:100%; height:auto; }}
.energy-low {{ fill:#7b9fc4; }} .energy-medium {{ fill:#d6ad55; }} .energy-high {{ fill:#c97878; }}
.beat {{ stroke:#7b9fc4; stroke-width:1; opacity:.7; }}
.downbeat {{ stroke:#e39a35; stroke-width:2.5; }}
.boundary {{ stroke:#aaa; stroke-width:1.5; stroke-dasharray:4 3; }}
.recommendation {{ stroke:#39a879; stroke-width:2.5; stroke-dasharray:5 3; }}
.clip-bar {{ fill:#688bb5; rx:5; }} .clip-label {{ fill:CanvasText; font-size:12px; }}
.tick {{ fill:GrayText; font-size:11px; }} .legend {{ display:flex; flex-wrap:wrap; gap:1rem; }}
audio {{ width:min(100%,700px); }} .note {{ color:GrayText; }}
</style></head><body>
<h1>Music & Timeline Sync Review</h1>
<p>{(f"{report.bpm:.1f} BPM" if report.bpm is not None else "Tempo undetected")} ·
beat confidence {report.beat_confidence:.2f} ·
{len(report.beat_times_seconds)} beats · {len(report.downbeat_times_seconds)} downbeats</p>
{audio_html}
<div class="legend"><span>■ Energy (low / medium / high)</span><span>│ Beat</span>
<span>│ Downbeat</span><span>┆ Current shot boundary</span>
<span>┆ Suggested boundary</span></div>
<div class="track"><svg viewBox="0 0 {width} {svg_height}" role="img"
aria-label="Soundtrack energy, beat markers, and shot timeline on a shared time axis">
{ticks}<text x="12" y="64" class="clip-label">Music energy</text>
{''.join(energy_rects)}{beats}{downbeats}
<text x="12" y="190" class="clip-label">Shot boundaries</text>
{''.join(boundary_rows)}{''.join(clip_rows)}
</svg></div>
<p class="note">All tracks share soundtrack time in seconds. Solid beat markers and stronger
downbeat markers are signal estimates; dashed vertical lines show current shot boundaries,
and green dashed lines show suggested anchors when available. Listen and inspect the footage:
beat detection can be imperfect, and not every cut should land on a beat.</p>
<h2>Boundary details</h2><ul>
{''.join(f'<li>{html.escape(item["previous_clip_id"])} → {html.escape(item["next_clip_id"])}: '
f'{item["boundary_seconds"]:.2f}s; beat delta '
f'{item["beat_delta_seconds"] if item["beat_delta_seconds"] is not None else "—"}; '
f'downbeat delta {item.get("downbeat_delta_seconds")}; '
f'suggestion {item["recommended_anchor"]} '
f'{item.get("recommended_boundary_seconds")}</li>'
for item in report.boundaries)}
</ul>
<p class="note">Read-only review. This page does not edit, approve, or render the timeline.
Use the existing <code>review-music-sync</code> CLI workflow to accept or reject proposed
timing changes; accepted changes create a separate unapproved timeline revision.</p>
<script>const previewData = {payload};</script>
</body></html>"""


def save_music_timeline_preview(
    folder: Path,
    report: MusicTimelinePreview,
    *,
    output_path: Path | None = None,
) -> tuple[Path, Path]:
    """Save a version-bound JSON report and HTML preview without overwriting prior output."""
    root = folder.expanduser().resolve()
    base = (
        output_path or root / ".video-agent" / "music-timeline-preview" / "preview"
    ).expanduser()
    base = (root / base).resolve() if not base.is_absolute() else base.resolve()
    try:
        base.relative_to(root)
    except ValueError as exc:
        raise MusicTimelinePreviewError("Preview output must be inside the project folder") from exc
    json_path, html_path = base.with_suffix(".json"), base.with_suffix(".html")
    if json_path.exists() or html_path.exists():
        raise FileExistsError(f"Music timeline preview already exists: {base}")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path = Path(report.music_path)
    audio_source = quote_path(os.path.relpath(audio_path, html_path.parent))
    html_path.write_text(
        render_music_timeline_preview_html(report, audio_source=audio_source), encoding="utf-8"
    )
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    register_artifact(
        root,
        json_path,
        dependencies=[root / report.timeline_path, root / report.music_analysis_path],
    )
    return json_path, html_path


def quote_path(value: str) -> str:
    """URL-encode a project-relative local media path."""
    from urllib.parse import quote

    return quote(value.replace(os.sep, "/"), safe="/:@-._~")
