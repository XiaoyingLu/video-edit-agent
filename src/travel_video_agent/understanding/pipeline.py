"""Orchestration for local, scene-level video understanding."""

import hashlib
from pathlib import Path

from ..media import analyze_folder, discover_videos
from .frame_sampler import extract_representative_frames
from .scene_detection import detect_scenes
from .schemas import FrameObservation, VideoScene, VideoUnderstanding
from .visual_analyzer import OllamaVisionAnalyzer


def _scene_summary(observations: list[FrameObservation]) -> str:
    descriptions = [item.description for item in observations]
    return (
        " ".join(descriptions) if descriptions else "No representative frames could be extracted."
    )


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
            scenes.append(
                VideoScene(
                    scene_id=f"{video_key}-scene-{scene_index:04d}",
                    start_seconds=start,
                    end_seconds=end,
                    summary=_scene_summary(observations),
                    observations=observations,
                )
            )

    return VideoUnderstanding(
        project_path=str(root),
        video_count=media.video_count,
        scene_count=len(scenes),
        scenes=scenes,
        model=model,
    )
