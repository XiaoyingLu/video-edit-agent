"""Visual feature extraction and deterministic highlight candidate ranking."""

import hashlib
from pathlib import Path

import cv2
import numpy as np

from ..artifact_tracking import validate_artifact
from ..media import analyze_folder, discover_videos
from ..understanding.frame_sampler import extract_representative_frames
from ..understanding.scene_detection import detect_scenes
from ..understanding.schemas import VideoUnderstanding
from .schemas import HighlightCandidate, HighlightSignals, ProjectHighlights


def _frame_metrics(frame: np.ndarray) -> tuple[float, float, float]:
    """Return normalized sharpness, exposure quality, and saturation."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 500.0, 1.0)
    exposure = float(np.logical_and(gray >= 20, gray <= 235).mean())
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = float(hsv[:, :, 1].mean()) / 255.0
    return sharpness, exposure, saturation


def _measure_frames(frame_paths: list[Path]) -> HighlightSignals:
    """Measure visual quality and inter-frame motion from sampled JPEGs."""
    if not frame_paths:
        raise ValueError("No frames were extracted for a highlight candidate.")

    frames: list[np.ndarray] = []
    metrics: list[tuple[float, float, float]] = []
    for path in frame_paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        frames.append(frame)
        metrics.append(_frame_metrics(frame))
    if not frames:
        raise ValueError("Could not read any sampled frames.")

    motion_values: list[float] = []
    for previous, current in zip(frames, frames[1:], strict=False):
        previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_values.append(min(float(np.mean(magnitude)) / 8.0, 1.0))

    count = len(metrics)
    return HighlightSignals(
        motion=sum(motion_values) / len(motion_values) if motion_values else 0.0,
        sharpness=sum(item[0] for item in metrics) / count,
        exposure=sum(item[1] for item in metrics) / count,
        saturation=sum(item[2] for item in metrics) / count,
    )


def _score(signals: HighlightSignals) -> tuple[float, list[str]]:
    """Apply a transparent weighted score; motion is a cue, not a requirement."""
    weights = {"motion": 0.25, "sharpness": 0.35, "exposure": 0.25, "saturation": 0.15}
    values = signals.model_dump()
    score = sum(values[name] * weight for name, weight in weights.items()) * 100
    reasons = [
        f"{name.title()} contributed {values[name] * weight * 100:.1f}/{weight * 100:.0f} points"
        for name, weight in weights.items()
    ]
    if signals.motion < 0.08:
        reasons.append("Low motion: this may still be a valuable calm scenic shot.")
    return round(score, 1), reasons


def find_highlights_folder(
    folder: Path,
    *,
    samples_per_scene: int = 3,
    understanding_path: Path | None = None,
) -> ProjectHighlights:
    """Build ranked, review-only highlight candidates from each detected scene."""
    if not 1 <= samples_per_scene <= 10:
        raise ValueError("samples_per_scene must be between 1 and 10")

    root = folder.expanduser().resolve()
    media = analyze_folder(root)
    artifact_dir = root / ".video-agent"
    source_understanding = understanding_path or artifact_dir / "understanding.json"
    understanding: VideoUnderstanding | None = None
    if source_understanding.exists():
        validate_artifact(root, source_understanding)
        understanding = VideoUnderstanding.model_validate_json(
            source_understanding.read_text(encoding="utf-8")
        )

    scenes_by_video: dict[str, list] = {}
    if understanding is not None:
        for scene in understanding.scenes:
            video_key = scene.scene_id.split("-scene-", maxsplit=1)[0]
            scenes_by_video.setdefault(video_key, []).append(scene)

    candidates: list[HighlightCandidate] = []
    for video_path in discover_videos(root):
        relative_path = video_path.relative_to(root).as_posix()
        video_info = next(item for item in media.videos if item.path == relative_path)
        video_key = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]
        known_scenes = scenes_by_video.get(video_key, [])
        intervals = (
            [(scene.start_seconds, scene.end_seconds, scene) for scene in known_scenes]
            if known_scenes
            else [
                (start, end, None)
                for start, end in detect_scenes(video_path, video_info.duration_seconds)
            ]
        )
        for index, (start, end, scene) in enumerate(intervals, start=1):
            if end <= start:
                continue
            frame_paths = (
                [root / observation.frame_path for observation in scene.observations]
                if scene is not None
                else []
            )
            if len(frame_paths) < 2 or not all(path.is_file() for path in frame_paths):
                frame_dir = artifact_dir / "highlight-frames" / video_key / f"scene-{index:04d}"
                sampled = extract_representative_frames(
                    video_path, start, end, frame_dir, samples_per_scene=samples_per_scene
                )
                frame_paths = [path for _, path in sampled]
            if not frame_paths:
                continue
            signals = _measure_frames(frame_paths)
            score, reasons = _score(signals)
            candidates.append(
                HighlightCandidate(
                    candidate_id=f"{video_key}-highlight-{index:04d}",
                    video_path=relative_path,
                    start_seconds=round(start, 3),
                    end_seconds=round(end, 3),
                    duration_seconds=round(end - start, 3),
                    score=score,
                    signals=signals,
                    reasons=reasons,
                    scene_summary=scene.summary if scene is not None else None,
                )
            )

    candidates.sort(key=lambda item: (-item.score, item.video_path, item.start_seconds))
    return ProjectHighlights(
        project_path=str(root),
        video_count=media.video_count,
        candidate_count=len(candidates),
        candidates=candidates,
    )
