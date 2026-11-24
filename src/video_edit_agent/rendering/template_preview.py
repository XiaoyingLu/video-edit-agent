"""Non-destructive visual previews for reusable editing templates."""

from __future__ import annotations

import html
import json
import shutil
import tempfile
from pathlib import Path

from ..artifact_tracking import validate_artifact
from ..editing.planning_schemas import VideoProjectPlan
from ..editing.timeline_schemas import ProjectDraftTimeline
from ..media_runner import MediaError, run_ffmpeg
from ..templates import load_template
from .overlay_filters import _append_title_overlay_filters
from .title_overlays import TitleOverlay, _template_overlay_style


class TemplatePreviewError(MediaError):
    """Raised when a template preview cannot be safely rendered."""


def _probe_dimensions(ffprobe: str, path: Path) -> tuple[int, int]:
    result = run_ffmpeg(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        timeout=30,
        label=f"ffprobe while inspecting preview frame {path}",
        error_type=TemplatePreviewError,
    )
    try:
        streams = json.loads(result.stdout).get("streams", [])
        width = int(streams[0]["width"])
        height = int(streams[0]["height"])
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TemplatePreviewError(f"could not determine preview frame dimensions: {path}") from exc
    if width < 2 or height < 2:
        raise TemplatePreviewError(f"preview frame has invalid dimensions: {path}")
    return width, height


def _opening_overlays(template, treatment) -> list[TitleOverlay]:
    """Build in-memory opening overlays using the same safe styles as title proposals."""
    values = {
        "{title}": treatment.working_title.strip(),
        "{logline}": treatment.logline.strip(),
        "{closing_line}": "",
        "{date}": "",
        "{location}": "",
    }
    limits = {
        "{title}": 40,
        "{logline}": 48,
        "{closing_line}": 40,
        "{date}": 40,
        "{location}": 40,
    }
    opening_items = template.sequences.get("opening", [])
    primary_item = next(
        (item for item in opening_items if item.slot == "{title}"),
        opening_items[0] if opening_items else None,
    )
    primary_placement = (
        template.title_styles[primary_item.style].placement
        if primary_item is not None
        else "center"
    )
    overlays: list[TitleOverlay] = []
    for index, item in enumerate(opening_items, start=1):
        text = values[item.slot]
        if not text:
            continue
        limit = limits[item.slot]
        if len(text) > limit:
            text = text[: limit - 3].rstrip() + "..."
        style = _template_overlay_style(template, item.style)
        style["animation"] = "none"
        if item.slot == "{logline}":
            style["placement"] = "bottom" if primary_placement != "bottom" else "center"
        overlays.append(
            TitleOverlay(
                overlay_id=f"preview-opening-{index}",
                text=text,
                start_seconds=0,
                end_seconds=5,
                exit_animation="none",
                sequence_role="opening",
                decision="approved",
                **style,
            )
        )
    if not overlays:
        raise ValueError(f"template {template.id!r} has no usable opening title slots")
    return overlays


def render_template_previews(
    folder: Path,
    template_ids: list[str],
    output_dir: Path | None = None,
) -> tuple[Path, list[Path], float]:
    """Render two or three template opening cards over one sampled approved shot."""
    if len(template_ids) not in {2, 3}:
        raise ValueError("compare exactly two or three template IDs")
    if len(set(template_ids)) != len(template_ids):
        raise ValueError("template IDs must be unique")

    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project folder does not exist: {root}")
    destination = (output_dir or root / ".video-agent" / "template-previews").expanduser().resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("template preview output must be inside the project folder") from exc

    timeline_file = root / ".video-agent" / "optimized-timeline.json"
    plan_file = root / ".video-agent" / "video-project-plan.json"
    for artifact, label in (
        (timeline_file, "optimized timeline"),
        (plan_file, "video project plan"),
    ):
        if not artifact.is_file():
            raise FileNotFoundError(f"Missing {label}: {artifact}")
        validate_artifact(root, artifact)

    timeline = ProjectDraftTimeline.model_validate_json(timeline_file.read_text(encoding="utf-8"))
    plan = VideoProjectPlan.model_validate_json(plan_file.read_text(encoding="utf-8"))
    if Path(timeline.project_path).resolve() != root or Path(plan.project_path).resolve() != root:
        raise ValueError("preview inputs belong to a different project folder")
    if not timeline.timeline_approved:
        raise ValueError("the optimized timeline must be approved before previewing templates")
    if plan.story_treatment is None:
        raise ValueError("the video project plan must contain a story treatment")

    clip = next((item for item in timeline.clips if item.decision == "approved"), None)
    if clip is None:
        raise ValueError("the approved timeline contains no approved clips")
    relative_source = Path(clip.video_path)
    if relative_source.is_absolute():
        raise ValueError("timeline source path must be project-relative")
    source = (root / relative_source).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ValueError("timeline source escapes the project folder") from exc
    if not source.is_file():
        raise FileNotFoundError(f"timeline source video does not exist: {source}")
    timestamp = (
        clip.source_start_seconds + (clip.source_end_seconds - clip.source_start_seconds) / 2
    )

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise TemplatePreviewError("ffmpeg and ffprobe must be installed and available on PATH")

    templates = [(template_id, *load_template(template_id, root)) for template_id in template_ids]
    overlays_by_template = [
        _opening_overlays(template, plan.story_treatment) for _, template, _ in templates
    ]
    output_files = [destination / f"{template_id}.jpg" for template_id in template_ids]
    index_file = destination / "index.html"
    if any(path.exists() for path in [*output_files, index_file]):
        raise FileExistsError(
            f"Template preview output already exists in {destination}; choose another --output-dir."
        )

    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="video-agent-template-preview-") as temporary:
        frame = Path(temporary) / "sample.jpg"
        run_ffmpeg(
            [
                ffmpeg,
                "-hide_banner",
                "-nostdin",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-vf",
                "scale=1280:-2:force_original_aspect_ratio=decrease",
                "-q:v",
                "2",
                str(frame),
            ],
            timeout=90,
            label=f"Extracting template preview frame from {source}",
            error_type=TemplatePreviewError,
        )
        if not frame.is_file() or frame.stat().st_size == 0:
            raise TemplatePreviewError("FFmpeg did not produce a preview frame")
        width, height = _probe_dimensions(ffprobe, frame)

        for (template_id, _template, _source_path), overlays, output in zip(
            templates, overlays_by_template, output_files, strict=True
        ):
            filters: list[str] = []
            output_label = _append_title_overlay_filters(filters, overlays, width, height, "0:v")
            run_ffmpeg(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-nostdin",
                    "-y",
                    "-loop",
                    "1",
                    "-framerate",
                    "1",
                    "-i",
                    str(frame),
                    "-filter_complex",
                    ";".join(filters),
                    "-map",
                    f"[{output_label}]",
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(output),
                ],
                timeout=90,
                label=f"Rendering {template_id} opening-card preview",
                error_type=TemplatePreviewError,
            )
            if not output.is_file() or output.stat().st_size == 0:
                raise TemplatePreviewError(f"FFmpeg did not produce {template_id} preview")

    cards = []
    for template_id, template, source_path in templates:
        cards.append(
            "<article><h2>"
            + html.escape(template_id)
            + f" <small>v{template.version}</small></h2>"
            + (
                f'<img src="{html.escape(template_id)}.jpg" '
                f'alt="{html.escape(template_id)} opening card">'
            )
            + f"<p>Source: {html.escape(str(source_path))}</p></article>"
        )
    index_file.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Template preview comparison</title>"
        "<style>body{font:16px system-ui;margin:2rem;background:#171717;color:#f5f5f5}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}"
        "article{background:#292929;padding:1rem;border-radius:12px;min-width:0}"
        "img{display:block;width:100%;height:auto}small,p{color:#bbb;font-size:.8rem}</style>"
        "<h1>Template preview comparison</h1><p>Same approved footage frame and opening text; "
        "visual styling varies by template. Previews do not approve or apply templates.</p>"
        '<main class="grid">' + "".join(cards) + "</main></html>",
        encoding="utf-8",
    )
    return index_file, output_files, timestamp
