"""Generate a portable HyperFrames composition over an approved FFmpeg draft."""

from __future__ import annotations

import html
import json
import math
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..artifact_tracking import StaleArtifactError, register_artifact, validate_artifact
from ..editing.timeline_schemas import ProjectDraftTimeline


class MotionGraphicsError(RuntimeError):
    """Raised when a motion-graphics project cannot be generated safely."""


class MotionGraphicsReport(BaseModel):
    """Summary of the generated optional HyperFrames project."""

    model_config = ConfigDict(extra="forbid")

    project_directory: str
    composition_path: str
    media_path: str
    duration_seconds: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    has_audio: bool
    title: str
    location: str | None = None


def _inside(root: Path, path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise MotionGraphicsError(f"{label} must be inside the project folder: {resolved}") from exc
    return resolved


def _probe_video(path: Path) -> tuple[float, int, int, bool]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,width,height",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, check=False, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise MotionGraphicsError(f"Could not inspect rendered video with ffprobe: {exc}") from exc
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace")[-1000:]
        raise MotionGraphicsError(f"ffprobe could not inspect {path.name}: {detail}")
    try:
        payload = json.loads(result.stdout)
        duration = float(payload["format"]["duration"])
        streams = payload.get("streams", [])
        video = next(stream for stream in streams if stream["codec_type"] == "video")
        width, height = int(video["width"]), int(video["height"])
        has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    except (KeyError, StopIteration, TypeError, ValueError) as exc:
        raise MotionGraphicsError(
            "Rendered draft has invalid or incomplete video metadata."
        ) from exc
    if not math.isfinite(duration) or duration <= 0 or width <= 0 or height <= 0:
        raise MotionGraphicsError("Rendered draft must have a positive duration and dimensions.")
    return duration, width, height, has_audio


def _composition_html(
    *,
    duration: float,
    width: int,
    height: int,
    title: str,
    location: str | None,
    has_audio: bool,
) -> str:
    title_duration = min(5.0, duration)
    location_start = min(0.9, duration * 0.18)
    location_duration = max(0.1, min(5.0, duration - location_start))
    safe_title = html.escape(title, quote=True)
    safe_location = html.escape(location or "", quote=True)
    audio_attributes = 'data-has-audio="true"' if has_audio else "muted"
    location_markup = ""
    location_animation = ""
    if location:
        location_markup = f"""
      <div id="location-card" class="clip location-card"
           data-start="{location_start:.3f}" data-duration="{location_duration:.3f}"
           data-track-index="3">
        <span class="location-rule"></span>
        <span class="location-label">{safe_location}</span>
      </div>"""
        location_animation = f"""
      timeline.fromTo("#location-card",
        {{ opacity: 0, y: 18 }},
        {{ opacity: 1, y: 0, duration: 0.55, ease: "power2.out" }},
        {location_start:.3f}
      );"""
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width={width}, height={height}" />
    <title>{safe_title}</title>
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
    <style>
      * {{ box-sizing: border-box; }}
      html, body {{
        margin: 0; width: {width}px; height: {height}px;
        overflow: hidden; background: #090b10;
      }}
      #root {{ position: relative; width: {width}px; height: {height}px; overflow: hidden; }}
      #draft-video {{
        position: absolute; inset: 0; width: 100%; height: 100%;
        object-fit: contain;
      }}
      .clip {{ position: absolute; inset: 0; pointer-events: none; }}
      .title-shade {{
        background: linear-gradient(
          180deg, rgba(5, 8, 14, .48), transparent 35%,
          transparent 62%, rgba(5, 8, 14, .35)
        );
        z-index: 2;
      }}
      .title-card {{
        z-index: 3; display: flex; align-items: center; justify-content: center;
        padding: 8% 10%; text-align: center;
      }}
      h1 {{
        max-width: 82%; margin: 0; color: #fff; font-family: Georgia, "Times New Roman", serif;
        font-size: clamp(48px, 6.2vw, 112px); font-weight: 500; line-height: 1.08;
        letter-spacing: .015em; text-shadow: 0 4px 30px rgba(0,0,0,.5);
      }}
      .location-card {{
        inset: auto auto 9% 7%; z-index: 4; display: flex; align-items: center;
        gap: 18px; width: auto; height: auto; color: white; font-family: Arial, sans-serif;
      }}
      .location-rule {{ width: 4px; height: 48px; background: #e9c98c; }}
      .location-label {{
        font-size: clamp(20px, 2.1vw, 36px); font-weight: 600; letter-spacing: .12em;
        text-transform: uppercase; text-shadow: 0 2px 14px rgba(0,0,0,.65);
      }}
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0"
         data-duration="{duration:.3f}" data-width="{width}" data-height="{height}">
      <video id="draft-video" src="./assets/rendered-draft.mp4"
             data-start="0" data-duration="{duration:.3f}" data-track-index="0"
             {audio_attributes} playsinline></video>
      <div id="title-shade" class="clip title-shade"
           data-start="0" data-duration="{duration:.3f}" data-track-index="1"></div>
      <section id="title-card" class="clip title-card"
               data-start="0" data-duration="{title_duration:.3f}" data-track-index="2">
        <h1 id="title">{safe_title}</h1>
      </section>{location_markup}
    </div>
    <script>
      window.__timelines = window.__timelines || {{}};
      const timeline = gsap.timeline({{ paused: true }});
      timeline.fromTo("#title",
        {{ opacity: 0, y: 42, letterSpacing: "0.08em" }},
        {{ opacity: 1, y: 0, letterSpacing: "0.015em", duration: 0.85, ease: "power3.out" }},
        0.15
      );
      {location_animation}
      window.__timelines.main = timeline;
    </script>
  </body>
</html>
"""


def create_hyperframes_project(
    folder: Path,
    *,
    title: str,
    location: str | None = None,
    timeline_path: Path | None = None,
    video_path: Path | None = None,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> MotionGraphicsReport:
    """Build a HyperFrames HTML composition over a previously rendered approved draft."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project folder does not exist: {root}")
    if not title.strip() or len(title) > 120:
        raise ValueError("Title must contain 1–120 characters.")
    if location is not None and (not location.strip() or len(location) > 100):
        raise ValueError("Location must contain 1–100 characters when supplied.")

    timeline_file = _inside(
        root,
        timeline_path or root / ".video-agent" / "optimized-timeline.json",
        "Timeline file",
    )
    video_file = _inside(
        root,
        video_path or root / ".video-agent" / "rendered-draft.mp4",
        "Rendered video",
    )
    destination = _inside(
        root,
        output_dir or root / ".video-agent" / "hyperframes-project",
        "Output directory",
    )
    if not timeline_file.is_file():
        raise FileNotFoundError(f"Approved timeline not found: {timeline_file}")
    if not video_file.is_file():
        raise FileNotFoundError(f"Rendered draft not found: {video_file}")

    try:
        validate_artifact(root, timeline_file)
        validate_artifact(root, video_file)
    except StaleArtifactError as exc:
        raise MotionGraphicsError(f"An upstream artifact is stale: {exc}") from exc

    try:
        timeline = ProjectDraftTimeline.model_validate_json(
            timeline_file.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise MotionGraphicsError(f"Could not load approved timeline: {exc}") from exc
    if not timeline.timeline_approved:
        raise MotionGraphicsError("Shot timeline is not approved; run review-timeline first.")
    if any(clip.decision == "pending" for clip in timeline.clips):
        raise MotionGraphicsError("Shot timeline still contains pending clips.")
    if not any(clip.decision == "approved" for clip in timeline.clips):
        raise MotionGraphicsError("Shot timeline has no approved clips.")

    duration, width, height, has_audio = _probe_video(video_file)
    assets_dir = destination / "assets"
    composition = destination / "index.html"
    metadata = destination / "meta.json"
    media = assets_dir / "rendered-draft.mp4"
    existing = [path for path in (composition, metadata, media) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"HyperFrames project already exists at {destination}; "
            "use --overwrite to replace generated files."
        )

    destination.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    if media.exists() or media.is_symlink():
        media.unlink()
    shutil.copy2(video_file, media)

    composition.write_text(
        _composition_html(
            duration=duration,
            width=width,
            height=height,
            title=title.strip(),
            location=location.strip() if location else None,
            has_audio=has_audio,
        ),
        encoding="utf-8",
    )
    metadata.write_text(
        json.dumps(
            {
                "name": title.strip(),
                "id": "video-agent-motion-graphics",
                "width": width,
                "height": height,
                "fps": 30,
                "duration_seconds": round(duration, 3),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    register_artifact(
        root,
        composition,
        dependencies=[timeline_file, video_file],
        config={"title": title.strip(), "location": location},
    )
    return MotionGraphicsReport(
        project_directory=str(destination),
        composition_path=str(composition),
        media_path=str(media),
        duration_seconds=round(duration, 3),
        width=width,
        height=height,
        has_audio=has_audio,
        title=title.strip(),
        location=location.strip() if location else None,
    )
