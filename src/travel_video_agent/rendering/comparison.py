"""Side-by-side, read-only comparison of two rendered video drafts."""

import base64
import html
import json
import shutil
import tempfile
from pathlib import Path

import cv2
from pydantic import BaseModel, ConfigDict, Field

from ..artifact_tracking import validate_artifact
from .critique import (
    AudioQualitySummary,
    DraftCritiqueError,
    FrameQualitySample,
    _analyze_audio,
    _parse_frame_rate,
    _run,
)


class DraftComparisonError(RuntimeError):
    """Raised when two rendered drafts cannot be compared safely."""


class DraftVideoMetadata(BaseModel):
    """Comparable media properties from one rendered draft."""

    model_config = ConfigDict(extra="forbid")

    path: str
    duration_seconds: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_rate: float | None = Field(default=None, gt=0)
    video_codec: str
    audio_codec: str | None = None
    audio: AudioQualitySummary


class ComparisonFrameSample(BaseModel):
    """Metrics and visual evidence at a normalized position in both drafts."""

    model_config = ConfigDict(extra="forbid")

    normalized_position: float = Field(ge=0, le=1)
    before: FrameQualitySample
    after: FrameQualitySample
    mean_luma_delta: float
    dark_pixel_ratio_delta: float
    bright_pixel_ratio_delta: float
    sharpness_variance_delta: float
    before_image_data_uri: str = Field(exclude=True)
    after_image_data_uri: str = Field(exclude=True)


class DraftComparisonReport(BaseModel):
    """Objective comparison report; it does not decide which edit is better."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    project_path: str
    before: DraftVideoMetadata
    after: DraftVideoMetadata
    duration_delta_seconds: float
    resolution_changed: bool
    frame_rate_delta: float | None = None
    audio_presence_changed: bool
    mean_volume_delta_db: float | None = None
    silence_duration_delta_seconds: float
    frames: list[ComparisonFrameSample] = Field(default_factory=list)
    review_required: bool = True
    timeline_modified: bool = False
    rendered_videos_modified: bool = False


def _project_video(root: Path, path: Path, label: str) -> Path:
    candidate = path.expanduser()
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DraftComparisonError(f"{label} must be inside the project folder.") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    if resolved.suffix.lower() != ".mp4":
        raise DraftComparisonError(f"{label} must be an MP4 file.")
    validate_artifact(root, resolved)
    return resolved


def _probe_video(ffprobe: str, ffmpeg: str, root: Path, video: Path) -> DraftVideoMetadata:
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate",
            "-of",
            "json",
            str(video),
        ],
        timeout=60,
    )
    if result.returncode != 0:
        raise DraftComparisonError(
            f"ffprobe could not inspect {video.name}: {result.stderr.strip()}"
        )
    try:
        payload = json.loads(result.stdout)
        duration = float(payload["format"]["duration"])
        streams = payload["streams"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise DraftComparisonError(f"ffprobe returned incomplete metadata for {video.name}.") from exc
    video_stream = next((item for item in streams if item.get("codec_type") == "video"), None)
    if video_stream is None:
        raise DraftComparisonError(f"{video.name} does not contain a video stream.")
    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    if duration <= 0 or width <= 0 or height <= 0:
        raise DraftComparisonError(f"{video.name} has invalid duration or dimensions.")
    audio_stream = next((item for item in streams if item.get("codec_type") == "audio"), None)
    try:
        audio = _analyze_audio(ffmpeg, video, audio_stream is not None)
    except DraftCritiqueError as exc:
        raise DraftComparisonError(str(exc)) from exc
    return DraftVideoMetadata(
        path=video.relative_to(root).as_posix(),
        duration_seconds=duration,
        width=width,
        height=height,
        frame_rate=_parse_frame_rate(video_stream.get("avg_frame_rate")),
        video_codec=str(video_stream.get("codec_name") or "unknown"),
        audio_codec=str(audio_stream.get("codec_name")) if audio_stream else None,
        audio=audio,
    )


def _extract_comparison_frame(
    ffmpeg: str, video: Path, timestamp: float, destination: Path
) -> tuple[FrameQualitySample, str]:
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
    if result.returncode != 0 or not destination.is_file():
        raise DraftComparisonError(
            f"Could not sample {video.name} at {timestamp:.2f}s: {result.stderr.strip()}"
        )
    image = cv2.imread(str(destination), cv2.IMREAD_COLOR)
    if image is None:
        raise DraftComparisonError(f"Could not decode a comparison frame from {video.name}.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    metrics = FrameQualitySample(
        timestamp_seconds=timestamp,
        mean_luma=float(gray.mean()),
        dark_pixel_ratio=float((gray < 16).mean()),
        bright_pixel_ratio=float((gray > 239).mean()),
        sharpness_variance=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    )
    encoded = base64.b64encode(destination.read_bytes()).decode("ascii")
    return metrics, f"data:image/jpeg;base64,{encoded}"


def compare_rendered_drafts(
    folder: Path,
    *,
    before_video: Path,
    after_video: Path,
    sample_count: int = 8,
) -> DraftComparisonReport:
    """Compare normalized frame samples and media metrics without modifying inputs."""
    if not 1 <= sample_count <= 16:
        raise ValueError("sample_count must be between 1 and 16")
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project folder does not exist: {root}")
    before_file = _project_video(root, before_video, "Before video")
    after_file = _project_video(root, after_video, "After video")
    if before_file == after_file:
        raise DraftComparisonError("Before and after videos must be different files.")

    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if ffprobe is None or ffmpeg is None:
        raise DraftComparisonError("ffmpeg and ffprobe must be installed and available on PATH.")
    before = _probe_video(ffprobe, ffmpeg, root, before_file)
    after = _probe_video(ffprobe, ffmpeg, root, after_file)

    samples: list[ComparisonFrameSample] = []
    with tempfile.TemporaryDirectory(prefix="video-agent-draft-comparison-") as temporary:
        directory = Path(temporary)
        for index in range(sample_count):
            position = (index + 1) / (sample_count + 1)
            before_time = before.duration_seconds * position
            after_time = after.duration_seconds * position
            before_metrics, before_image = _extract_comparison_frame(
                ffmpeg, before_file, before_time, directory / f"before-{index:02d}.jpg"
            )
            after_metrics, after_image = _extract_comparison_frame(
                ffmpeg, after_file, after_time, directory / f"after-{index:02d}.jpg"
            )
            samples.append(
                ComparisonFrameSample(
                    normalized_position=position,
                    before=before_metrics,
                    after=after_metrics,
                    mean_luma_delta=round(
                        after_metrics.mean_luma - before_metrics.mean_luma, 3
                    ),
                    dark_pixel_ratio_delta=round(
                        after_metrics.dark_pixel_ratio - before_metrics.dark_pixel_ratio, 4
                    ),
                    bright_pixel_ratio_delta=round(
                        after_metrics.bright_pixel_ratio - before_metrics.bright_pixel_ratio, 4
                    ),
                    sharpness_variance_delta=round(
                        after_metrics.sharpness_variance - before_metrics.sharpness_variance, 3
                    ),
                    before_image_data_uri=before_image,
                    after_image_data_uri=after_image,
                )
            )

    before_silence = sum(end - start for start, end in before.audio.silence_intervals)
    after_silence = sum(end - start for start, end in after.audio.silence_intervals)
    volume_delta = (
        after.audio.mean_volume_db - before.audio.mean_volume_db
        if before.audio.mean_volume_db is not None and after.audio.mean_volume_db is not None
        else None
    )
    frame_rate_delta = (
        after.frame_rate - before.frame_rate
        if before.frame_rate is not None and after.frame_rate is not None
        else None
    )
    return DraftComparisonReport(
        project_path=str(root),
        before=before,
        after=after,
        duration_delta_seconds=round(after.duration_seconds - before.duration_seconds, 3),
        resolution_changed=(before.width, before.height) != (after.width, after.height),
        frame_rate_delta=round(frame_rate_delta, 4) if frame_rate_delta is not None else None,
        audio_presence_changed=before.audio.has_audio != after.audio.has_audio,
        mean_volume_delta_db=round(volume_delta, 2) if volume_delta is not None else None,
        silence_duration_delta_seconds=round(after_silence - before_silence, 3),
        frames=samples,
    )


def render_comparison_html(report: DraftComparisonReport) -> str:
    """Build a self-contained HTML contact sheet with paired frame evidence."""
    rows = []
    for index, sample in enumerate(report.frames, start=1):
        before_image = sample.before_image_data_uri
        after_image = sample.after_image_data_uri
        rows.append(
            f"""
            <section class="sample">
              <h2>Sample {index} · {sample.normalized_position:.0%} through each draft</h2>
              <div class="pair">
                <figure><figcaption>Before · {sample.before.timestamp_seconds:.2f}s</figcaption>
                  <img src="{before_image}" alt="Before draft frame {index}"></figure>
                <figure><figcaption>After · {sample.after.timestamp_seconds:.2f}s</figcaption>
                  <img src="{after_image}" alt="After draft frame {index}"></figure>
              </div>
              <p class="metrics">Mean luma Δ {sample.mean_luma_delta:+.2f} ·
                Dark pixels Δ {sample.dark_pixel_ratio_delta:+.1%} ·
                Bright pixels Δ {sample.bright_pixel_ratio_delta:+.1%} ·
                Sharpness variance Δ {sample.sharpness_variance_delta:+.2f}</p>
            </section>"""
        )
    before_name = html.escape(report.before.path)
    after_name = html.escape(report.after.path)
    volume = (
        f"{report.mean_volume_delta_db:+.2f} dB"
        if report.mean_volume_delta_db is not None
        else "not comparable"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rendered Draft Comparison</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }}
h1 {{ margin-bottom: .25rem; }} .note {{ color: #777; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: .75rem; }}
.card {{ border: 1px solid #8886; border-radius: 10px; padding: .9rem; }}
.pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
figure {{ margin: 0; }} img {{ display: block; width: 100%; height: auto; border-radius: 8px; }}
figcaption {{ font-weight: 600; margin: .4rem 0; }} .sample {{ margin: 2rem 0; }}
.metrics {{ color: #777; font-size: .92rem; }}
@media(max-width:650px) {{ .pair {{ grid-template-columns: 1fr; }} }}
</style></head>
<body>
<h1>Rendered Draft Comparison</h1>
<p class="note">Matched by normalized playback position. Measurements are objective signals,
not a score of creative quality or a recommendation to keep either edit.</p>
<div class="summary">
  <div class="card"><strong>Before</strong><p>{before_name}</p>
    <p>{report.before.duration_seconds:.2f}s · {report.before.width}×{report.before.height}</p>
    <p>{html.escape(report.before.video_codec)} video · {html.escape(report.before.audio_codec or "no audio")}</p></div>
  <div class="card"><strong>After</strong><p>{after_name}</p>
    <p>{report.after.duration_seconds:.2f}s · {report.after.width}×{report.after.height}</p>
    <p>{html.escape(report.after.video_codec)} video · {html.escape(report.after.audio_codec or "no audio")}</p></div>
  <div class="card"><strong>Duration change</strong><p>{report.duration_delta_seconds:+.2f}s</p>
    <p>Resolution changed: {"yes" if report.resolution_changed else "no"}</p></div>
  <div class="card"><strong>Audio change</strong><p>Mean volume: {volume}</p>
    <p>Silence duration: {report.silence_duration_delta_seconds:+.2f}s</p></div>
</div>
{"".join(rows)}
<p class="note">Review the full videos and listen to the soundtrack before deciding.
No timeline or media files were modified.</p>
</body></html>
"""
