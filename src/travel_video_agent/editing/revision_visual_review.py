"""Visual, read-only before/after review for timeline revisions."""

import base64
import html
import json
import os
import shutil
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..artifact_tracking import register_artifact, validate_artifact
from ..rendering.critique import DraftCritiqueError, _run
from .revisions import TimelineRevisionError, compare_timeline_files
from .timeline_schemas import ProjectDraftTimeline


class RevisionFramePair(BaseModel):
    """Paired source frames for one changed timeline shot."""

    model_config = ConfigDict(extra="forbid")

    clip_id: str
    before_video_path: str
    after_video_path: str
    before_start_seconds: float
    before_end_seconds: float
    after_start_seconds: float
    after_end_seconds: float
    before_frame_data_uri: str = Field(default="", exclude=True)
    after_frame_data_uri: str = Field(default="", exclude=True)


class TimelineRevisionVisualReport(BaseModel):
    """Serializable visual review report; image bytes are kept in the HTML only."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    project_path: str
    before_timeline_path: str
    after_timeline_path: str
    before_duration_seconds: float
    after_duration_seconds: float
    duration_delta_seconds: float
    frames: list[RevisionFramePair] = Field(default_factory=list)
    review_required: bool = True
    timeline_modified: bool = False
    rendered_video_modified: bool = False


class TimelineRevisionVisualReviewError(ValueError):
    """Raised when visual revision evidence cannot be generated safely."""


def _project_media(root: Path, value: str) -> Path:
    candidate = Path(value)
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise TimelineRevisionVisualReviewError(
            "Timeline source media must remain inside the project folder"
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing timeline source media: {resolved}")
    return resolved


def _extract_frame(ffmpeg: str, video: Path, timestamp: float, destination: Path) -> str:
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
        raise TimelineRevisionVisualReviewError(str(exc)) from exc
    if result.returncode != 0 or not destination.is_file():
        raise TimelineRevisionVisualReviewError(
            f"Could not extract a frame from {video.name} at {timestamp:.2f}s: "
            f"{result.stderr.strip()}"
        )
    encoded = base64.b64encode(destination.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_timeline_revision_visual_review(
    folder: Path,
    *,
    before_path: Path,
    after_path: Path,
) -> TimelineRevisionVisualReport:
    """Extract midpoint source frames for each changed shot without editing media."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project folder does not exist: {root}")
    comparison = compare_timeline_files(root, before_path, after_path)
    if comparison.before_path == comparison.after_path:
        raise TimelineRevisionVisualReviewError(
            "Before and after timelines must be different files"
        )
    before_file = root / comparison.before_path
    after_file = root / comparison.after_path
    validate_artifact(root, before_file)
    validate_artifact(root, after_file)
    before = ProjectDraftTimeline.model_validate_json(before_file.read_text(encoding="utf-8"))
    after = ProjectDraftTimeline.model_validate_json(after_file.read_text(encoding="utf-8"))
    before_clips = {clip.clip_id: clip for clip in before.clips}
    after_clips = {clip.clip_id: clip for clip in after.clips}
    ffmpeg = shutil.which("ffmpeg") if comparison.changes else None
    if comparison.changes and ffmpeg is None:
        raise TimelineRevisionVisualReviewError(
            "ffmpeg must be installed and available on PATH"
        )

    frames: list[RevisionFramePair] = []
    with tempfile.TemporaryDirectory(prefix="video-agent-revision-review-") as temporary:
        directory = Path(temporary)
        for index, change in enumerate(comparison.changes):
            old = before_clips[change.clip_id]
            new = after_clips[change.clip_id]
            before_video = _project_media(root, old.video_path)
            after_video = _project_media(root, new.video_path)
            before_time = (old.source_start_seconds + old.source_end_seconds) / 2
            after_time = (new.source_start_seconds + new.source_end_seconds) / 2
            before_image = _extract_frame(
                ffmpeg, before_video, before_time, directory / f"before-{index:03}.jpg"
            )
            after_image = _extract_frame(
                ffmpeg, after_video, after_time, directory / f"after-{index:03}.jpg"
            )
            frames.append(
                RevisionFramePair(
                    clip_id=change.clip_id,
                    before_video_path=old.video_path,
                    after_video_path=new.video_path,
                    before_start_seconds=old.source_start_seconds,
                    before_end_seconds=old.source_end_seconds,
                    after_start_seconds=new.source_start_seconds,
                    after_end_seconds=new.source_end_seconds,
                    before_frame_data_uri=before_image,
                    after_frame_data_uri=after_image,
                )
            )

    return TimelineRevisionVisualReport(
        project_path=str(root),
        before_timeline_path=comparison.before_path,
        after_timeline_path=comparison.after_path,
        before_duration_seconds=comparison.before_duration_seconds,
        after_duration_seconds=comparison.after_duration_seconds,
        duration_delta_seconds=comparison.duration_delta_seconds,
        frames=frames,
    )


def render_timeline_revision_review_html(report: TimelineRevisionVisualReport) -> str:
    """Render self-contained, paired frame evidence for a local HTML review."""
    sections = []
    for frame in report.frames:
        sections.append(
            f"""
<section class="shot">
  <h2>{html.escape(frame.clip_id)}</h2>
  <div class="pair">
    <figure><figcaption>Before · {html.escape(frame.before_video_path)} ·
      {frame.before_start_seconds:.2f}–{frame.before_end_seconds:.2f}s</figcaption>
      <img src="{frame.before_frame_data_uri}" alt="Before frame for {html.escape(frame.clip_id)}">
    </figure>
    <figure><figcaption>After · {html.escape(frame.after_video_path)} ·
      {frame.after_start_seconds:.2f}–{frame.after_end_seconds:.2f}s</figcaption>
      <img src="{frame.after_frame_data_uri}" alt="After frame for {html.escape(frame.clip_id)}">
    </figure>
  </div>
</section>"""
        )
    if not sections:
        sections.append(
            "<p>No source intervals or transitions changed between these timelines.</p>"
        )
    delta = f"{report.duration_delta_seconds:+.2f}s"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Timeline Revision Visual Review</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui,sans-serif; }}
body {{ max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }}
header {{ border-bottom: 1px solid #8886; padding-bottom: 1rem; margin-bottom: 2rem; }}
.meta {{ color: #888; }} .shot {{ margin: 2rem 0; }}
.pair {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; }}
figure {{ margin:0; }} img {{ width:100%; height:auto; display:block; border-radius:8px; }}
figcaption {{ font-size:.9rem; margin:.5rem 0; overflow-wrap:anywhere; }}
.note {{ border:1px solid #8886; border-radius:8px; padding:1rem; }}
@media(max-width:650px) {{ .pair {{ grid-template-columns:1fr; }} }}
</style></head><body>
<header><h1>Timeline Revision Visual Review</h1>
<p class="meta">{len(report.frames)} changed shot(s) · Duration
{report.before_duration_seconds:.2f}s → {report.after_duration_seconds:.2f}s ({delta})</p>
<p>Each pair shows the midpoint of the source interval used before and after the revision.
This is a source-frame inspection aid, not a rendered playback comparison.</p></header>
{''.join(sections)}
<div class="note">Review the full shot context and transitions before approving the revised
timeline. This report does not approve, modify, or render either timeline.</div>
</body></html>"""


def save_timeline_revision_visual_review(
    folder: Path,
    report: TimelineRevisionVisualReport,
    *,
    output_path: Path | None = None,
) -> tuple[Path, Path]:
    """Save JSON metadata and a self-contained HTML visual review without overwriting."""
    root = folder.expanduser().resolve()
    base = (
        output_path or root / ".video-agent" / "timeline-revision-review" / "review"
    ).expanduser()
    base = (root / base).resolve() if not base.is_absolute() else base.resolve()
    try:
        base.relative_to(root)
    except ValueError as exc:
        raise TimelineRevisionVisualReviewError(
            "Review output must remain inside the project folder"
        ) from exc
    json_file = base.with_suffix(".json")
    html_file = base.with_suffix(".html")
    if json_file.exists() or html_file.exists():
        raise FileExistsError(f"Refusing to overwrite revision review: {base}")
    json_file.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.model_dump(mode="json"), indent=2) + "\n"
    temporary = json_file.with_suffix(".json.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, json_file)
    html_file.write_text(render_timeline_revision_review_html(report), encoding="utf-8")
    before_file = root / report.before_timeline_path
    after_file = root / report.after_timeline_path
    register_artifact(root, json_file, dependencies=[before_file, after_file])
    register_artifact(root, html_file, dependencies=[before_file, after_file, json_file])
    return json_file, html_file
