"""Orchestration for local, scene-level video understanding."""

import hashlib
import warnings
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from ..media import analyze_folder, discover_videos
from .frame_sampler import extract_representative_frames
from .scene_detection import detect_scenes
from .schemas import FrameObservation, SceneDynamics, VideoScene, VideoUnderstanding
from .visual_analyzer import OllamaVisionAnalyzer, VisionModelError


def _dominant_label(observations: list[FrameObservation], field: str) -> str:
    """Return the most common non-unknown editorial label, preserving first-seen ties."""
    labels = [getattr(item, field) for item in observations]
    counts = Counter(label for label in labels if label != "unknown")
    return counts.most_common(1)[0][0] if counts else "unknown"


def _scene_summary(observations: list[FrameObservation]) -> str:
    """Combine concise frame descriptions with grounded editorial context."""
    descriptions = [item.description for item in observations]
    if not descriptions:
        return "No representative frames could be extracted."
    unique_descriptions = list(dict.fromkeys(descriptions))
    return " ".join(unique_descriptions)


def understand_folder(
    folder: Path,
    model: str = "qwen2.5vl:7b",
    samples_per_scene: int = 3,
) -> VideoUnderstanding:
    """Detect scenes, sample frames, and describe them with local Ollama vision."""
    root = folder.expanduser().resolve()
    media = analyze_folder(root)
    analyzer = OllamaVisionAnalyzer(model=model)
    scenes: list[VideoScene] = []

    try:
        for video_path in discover_videos(root):
            relative_path = video_path.relative_to(root).as_posix()
            video_info = next(item for item in media.videos if item.path == relative_path)
            video_key = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]
            frame_root = root / ".video-agent" / "frames" / video_key
            intervals = detect_scenes(video_path, video_info.duration_seconds)
            for scene_index, (start, end) in enumerate(intervals, start=1):
                frame_dir = frame_root / f"scene-{scene_index:04d}"
                frames = extract_representative_frames(
                    video_path, start, end, frame_dir, samples_per_scene=samples_per_scene
                )
                observations: list[FrameObservation] = []
                for timestamp, frame_path in frames:
                    result = analyzer.analyze_frame(frame_path)
                    observations.append(
                        FrameObservation(
                            timestamp_seconds=timestamp,
                            frame_path=frame_path.relative_to(root).as_posix(),
                            **result,
                        )
                    )
                dynamics = None
                if len(frames) >= 2:
                    try:
                        dynamics = SceneDynamics.model_validate(analyzer.analyze_scene(frames))
                    except (VisionModelError, ValidationError) as exc:
                        warnings.warn(
                            f"Temporal analysis failed for scene {video_key}-scene-{scene_index:04d}: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                scenes.append(
                    VideoScene(
                        scene_id=f"{video_key}-scene-{scene_index:04d}",
                        start_seconds=start,
                        end_seconds=end,
                        summary=_scene_summary(observations),
                        story_function=_dominant_label(observations, "visual_role"),
                        dominant_mood=_dominant_label(observations, "mood"),
                        observations=observations,
                        dynamics=dynamics,
                    )
                )

        return VideoUnderstanding(
            project_path=str(root),
            video_count=media.video_count,
            scene_count=len(scenes),
            scenes=scenes,
            model=model,
        )
    finally:
        if not analyzer.unload_model():
            warnings.warn(
                f"Could not unload Ollama model '{model}'; it may remain resident in memory.",
                RuntimeWarning,
                stacklevel=2,
            )
