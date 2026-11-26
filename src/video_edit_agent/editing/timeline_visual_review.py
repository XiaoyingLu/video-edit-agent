"""Browser-based visual review for a proposed shot timeline."""

import base64
import hashlib
import html
import json
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field

from ..artifact_tracking import register_artifact, validate_artifact
from ..atomic_io import write_json_atomic
from ..rendering.critique import DraftCritiqueError, _run
from ..rendering.social_crop_anchors import load_crop_anchors
from .timeline_schemas import ProjectDraftTimeline
from .user_overrides import apply_user_overrides


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
    caption: str | None = None
    gain_db: float | None = None
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
    crop_anchors: dict[str, tuple[float, float]] = Field(default_factory=dict)
    crop_anchor_warning: str | None = None


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
    timeline = apply_user_overrides(timeline)
    anchor_file = root / ".video-agent" / "social-crop-anchors.json"
    crop_anchor_warning = None
    crop_anchors = {}
    if anchor_file.is_file():
        try:
            crop_anchors = load_crop_anchors(root, timeline_file, timeline, anchor_file)
        except ValueError as exc:
            crop_anchor_warning = str(exc)
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
                    caption=clip.caption,
                    gain_db=clip.gain_db,
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
        crop_anchors=crop_anchors,
        crop_anchor_warning=crop_anchor_warning,
        clips=clips,
    )


def _render_timeline_review_card(
    clip: TimelineVisualReviewClip,
    report: TimelineVisualReviewReport,
    media_sources: dict[str, str] | None,
) -> str:
    """Render one escaped clip card for the offline review page."""
    image = clip.frame_data_uri
    anchor_x, anchor_y = report.crop_anchors.get(clip.clip_id, (0.5, 0.5))
    details = [
        ("Scene", clip.scene_summary or "Not available"),
        ("Action", clip.temporal_summary or "Not available"),
        ("Editorial role", f"{clip.visual_role} · {clip.shot_scale} · {clip.mood}"),
        ("Sequence rationale", clip.sequence_reason),
        ("Continuity", clip.continuity_note),
        ("Pacing", clip.pacing_note),
        ("Music", f"{clip.music_energy} · {clip.music_pacing_note}"),
        ("Incoming transition", f"{clip.transition_in} · {clip.transition_duration_seconds:.2f}s"),
        ("Transcript", clip.transcript_excerpt or "Not available"),
    ]
    detail_html = "".join(
        f"<p><strong>{html.escape(label)}:</strong> {html.escape(value)}</p>"
        for label, value in details
    )
    media_src = (media_sources or {}).get(clip.clip_id, "")
    approved_checked = "checked" if clip.decision == "approved" else ""
    rejected_checked = "checked" if clip.decision == "rejected" else ""
    pending_checked = "checked" if clip.decision == "pending" else ""
    has_crop_anchor = "true" if clip.clip_id in report.crop_anchors else "false"
    transition_types = (
        "cut",
        "crossfade",
        "dissolve",
        "fadeblack",
        "wipeleft",
        "slideleft",
        "circleopen",
    )
    transition_options = "".join(
        f'<option value="{name}" '
        f"{'selected' if clip.transition_in == name else ''}"
        f">{name.replace('_', ' ').title()}</option>"
        for name in transition_types
    )
    return (
        f'<article class="clip" data-clip-id="{html.escape(clip.clip_id, quote=True)}" '
        f'data-has-crop-anchor="{has_crop_anchor}" '
        f'data-video-src="{html.escape(media_src, quote=True)}" '
        f'data-source-start="{clip.source_start_seconds:.3f}" '
        f'data-source-end="{clip.source_end_seconds:.3f}">'
        f'<div class="image"><img src="{image}" alt="Representative frame for '
        f'{html.escape(clip.clip_id, quote=True)}"></div>'
        f'<div class="body"><section class="crop-editor" '
        f'aria-label="Vertical crop for {html.escape(clip.clip_id, quote=True)}">'
        f'<div class="crop-preview"><img class="crop-preview-image" '
        f'src="{image}" alt="Vertical crop preview"></div>'
        f'<label>Horizontal focal point <input class="crop-anchor" '
        f'data-axis="x" type="range" min="0" max="1" step="0.01" '
        f'value="{anchor_x:.2f}"> <output>{anchor_x:.2f}</output></label>'
        f'<label>Vertical focal point <input class="crop-anchor" '
        f'data-axis="y" type="range" min="0" max="1" step="0.01" '
        f'value="{anchor_y:.2f}"> <output>{anchor_y:.2f}</output></label>'
        f'</section><div class="heading"><h2>{clip.sequence_index}. '
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
        f'<div class="trim clip-overrides"><label>Shot caption <input class="clip-caption" '
        f'type="text" maxlength="500" value="{html.escape(clip.caption or "", quote=True)}"></label>'
        f'<label>Source gain (dB) <input class="clip-gain" type="number" min="-60" max="24" '
        f'step="0.5" value="{"" if clip.gain_db is None else clip.gain_db}"></label></div>'
        f'<div class="trim transition-editor"><label>Transition into this shot '
        f'<select class="transition-type">{transition_options}</select></label>'
        f'<label>Transition duration (s) <input class="transition-duration" type="number" '
        f'min="0" max="2" step="0.1" value="{clip.transition_duration_seconds:.2f}"></label></div>'
        f"{detail_html}<fieldset><legend>Decision</legend>"
        f'<label><input type="radio" name="decision-{clip.sequence_index}" value="approved" '
        f"{approved_checked}> Approve</label>"
        f'<label><input type="radio" name="decision-{clip.sequence_index}" value="rejected" '
        f"{rejected_checked}> Reject</label>"
        f'<label><input type="radio" name="decision-{clip.sequence_index}" value="pending" '
        f"{pending_checked}> Skip / pending</label>"
        f"</fieldset></div></article>"
    )


def render_timeline_visual_review_html(
    report: TimelineVisualReviewReport,
    *,
    media_sources: dict[str, str] | None = None,
    submission_url: str | None = None,
) -> str:
    """Render a browser timeline review, optionally with live edit submission."""
    payload_json = json.dumps(report.model_dump(mode="json"), ensure_ascii=True).replace(
        "<", "\\u003c"
    )
    crop_warning_html = (
        f'<p role="status">Existing crop anchors were not loaded: '
        f"{html.escape(report.crop_anchor_warning)}</p>"
        if report.crop_anchor_warning
        else ""
    )
    cards_html = "".join(
        _render_timeline_review_card(clip, report, media_sources) for clip in report.clips
    )
    template_path = Path(__file__).with_name("templates") / "timeline_visual_review.html"
    template = template_path.read_text(encoding="utf-8")
    return (
        template.replace("__CLIP_COUNT__", str(len(report.clips)))
        .replace("__PROPOSED_DURATION__", f"{report.proposed_duration_seconds:.2f}")
        .replace("__TARGET_DURATION__", f"{report.target_duration_seconds:.2f}")
        .replace("__CROP_WARNING_HTML__", crop_warning_html)
        .replace("__CLIP_CARDS__", cards_html or "<p>No clips are present in this timeline.</p>")
        .replace("__PAYLOAD_JSON__", payload_json)
        .replace("__SUBMISSION_URL__", submission_url or "")
        .replace(
            "__EDIT_ACTION_LABEL__",
            "Submit visual edits" if submission_url else "Export visual edits (.json)",
        )
    )


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
    write_json_atomic(json_file, report)
    media_sources = {
        clip.clip_id: quote(
            os.path.relpath(
                (root / clip.video_path).resolve()
                if not Path(clip.video_path).is_absolute()
                else Path(clip.video_path).resolve(),
                html_file.parent,
            ).replace(os.sep, "/"),
            safe="/:@-._~",
        )
        for clip in report.clips
    }
    html_file.write_text(
        render_timeline_visual_review_html(report, media_sources=media_sources),
        encoding="utf-8",
    )
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
