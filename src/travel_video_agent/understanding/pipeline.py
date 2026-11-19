"""Orchestration for local, resumable scene-level video understanding."""

import hashlib
import json
import os
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..media import analyze_folder, discover_videos
from .frame_sampler import extract_representative_frames
from .scene_detection import detect_scenes
from .schemas import FrameObservation, SceneDynamics, VideoScene, VideoUnderstanding
from .visual_analyzer import OllamaVisionAnalyzer, VisionModelError

_CHECKPOINT_VERSION = 1
_CHECKPOINT_NAME = "understanding-checkpoint.json"


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


def _source_fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _frame_fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _load_checkpoint(
    path: Path,
    *,
    root: Path,
    model: str,
    samples_per_scene: int,
    sources: dict[str, dict[str, int]],
    resume: bool,
) -> dict[str, Any]:
    """Load compatible progress, or initialize a fresh checkpoint."""
    empty: dict[str, Any] = {
        "version": _CHECKPOINT_VERSION,
        "project_path": str(root),
        "model": model,
        "samples_per_scene": samples_per_scene,
        "sources": sources,
        "scenes": {},
    }
    if not resume or not path.is_file():
        return empty
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Could not read understanding checkpoint {path}: {exc}. "
            "Move the checkpoint aside to start a fresh run."
        ) from exc
    expected = ("version", "project_path", "model", "samples_per_scene", "sources")
    if not isinstance(checkpoint, dict) or any(
        checkpoint.get(key) != empty[key] for key in expected
    ):
        warnings.warn(
            "Existing understanding checkpoint does not match the current project inputs "
            "or settings; starting a fresh understanding run.",
            RuntimeWarning,
            stacklevel=2,
        )
        return empty
    if not isinstance(checkpoint.get("scenes"), dict):
        raise ValueError(
            f"Understanding checkpoint {path} has an invalid scenes section. "
            "Move the checkpoint aside to start a fresh run."
        )
    return checkpoint


def _save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    """Atomically persist progress so interruption loses at most one model request."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def understand_folder(
    folder: Path,
    model: str = "qwen2.5vl:7b",
    samples_per_scene: int = 3,
    *,
    checkpoint_path: Path | None = None,
    resume: bool = True,
) -> VideoUnderstanding:
    """Detect scenes and describe frames, checkpointing each completed vision request.

    Existing compatible frame observations and temporal scene analysis are reused on reruns.
    Pass resume=False to intentionally discard prior progress and start over.
    """
    root = folder.expanduser().resolve()
    media = analyze_folder(root)
    videos = discover_videos(root)
    sources = {
        path.relative_to(root).as_posix(): _source_fingerprint(path) for path in videos
    }
    checkpoint_file = (
        checkpoint_path.expanduser().resolve()
        if checkpoint_path is not None
        else root / ".video-agent" / _CHECKPOINT_NAME
    )
    checkpoint = _load_checkpoint(
        checkpoint_file,
        root=root,
        model=model,
        samples_per_scene=samples_per_scene,
        sources=sources,
        resume=resume,
    )
    analyzer = OllamaVisionAnalyzer(model=model)
    scenes: list[VideoScene] = []

    try:
        for video_path in videos:
            relative_path = video_path.relative_to(root).as_posix()
            video_info = next(item for item in media.videos if item.path == relative_path)
            video_key = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]
            frame_root = root / ".video-agent" / "frames" / video_key
            intervals = detect_scenes(video_path, video_info.duration_seconds)
            for scene_index, (start, end) in enumerate(intervals, start=1):
                scene_id = f"{video_key}-scene-{scene_index:04d}"
                frame_dir = frame_root / f"scene-{scene_index:04d}"
                frames = extract_representative_frames(
                    video_path, start, end, frame_dir, samples_per_scene=samples_per_scene
                )
                existing = checkpoint["scenes"].get(scene_id, {})
                if (
                    existing.get("video_path") != relative_path
                    or existing.get("start_seconds") != start
                    or existing.get("end_seconds") != end
                ):
                    existing = {
                        "video_path": relative_path,
                        "start_seconds": start,
                        "end_seconds": end,
                        "frames": {},
                        "dynamics": None,
                        "complete": False,
                    }
                    checkpoint["scenes"][scene_id] = existing

                observations_by_path: dict[str, FrameObservation] = {}
                for timestamp, frame_path in frames:
                    frame_key = frame_path.relative_to(root).as_posix()
                    fingerprint = _frame_fingerprint(frame_path)
                    cached = existing["frames"].get(frame_key)
                    if (
                        cached
                        and cached.get("timestamp_seconds") == timestamp
                        and cached.get("fingerprint") == fingerprint
                    ):
                        observation = FrameObservation.model_validate(cached["observation"])
                    else:
                        result = analyzer.analyze_frame(frame_path)
                        observation = FrameObservation(
                            timestamp_seconds=timestamp,
                            frame_path=frame_key,
                            **result,
                        )
                        existing["frames"][frame_key] = {
                            "timestamp_seconds": timestamp,
                            "fingerprint": fingerprint,
                            "observation": observation.model_dump(mode="json"),
                        }
                        _save_checkpoint(checkpoint_file, checkpoint)
                    observations_by_path[frame_key] = observation

                observations = [
                    observations_by_path[frame_path.relative_to(root).as_posix()]
                    for _, frame_path in frames
                ]
                dynamics = None
                if existing.get("complete"):
                    cached_dynamics = existing.get("dynamics")
                    if cached_dynamics is not None:
                        dynamics = SceneDynamics.model_validate(cached_dynamics)
                elif len(frames) >= 2:
                    try:
                        dynamics = SceneDynamics.model_validate(analyzer.analyze_scene(frames))
                    except (VisionModelError, ValidationError) as exc:
                        warnings.warn(
                            f"Temporal analysis failed for scene {scene_id}: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                existing["dynamics"] = (
                    dynamics.model_dump(mode="json") if dynamics is not None else None
                )
                existing["complete"] = True
                _save_checkpoint(checkpoint_file, checkpoint)

                scenes.append(
                    VideoScene(
                        scene_id=scene_id,
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
