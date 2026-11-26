"""Content-based scene boundary detection using PySceneDetect."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..media import MediaToolError


def detect_scenes(
    video_path: Path,
    duration_seconds: float,
    threshold: float = 27.0,
    min_scene_seconds: float = 1.0,
    show_progress: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[tuple[float, float]]:
    """Return scene intervals in seconds, falling back to the whole clip.

    When requested, report frame-level progress approximately once per second.
    The progress wrapper delegates all detection work to ContentDetector unchanged.
    """
    try:
        from scenedetect import ContentDetector, SceneManager, open_video
    except ImportError as exc:
        raise MediaToolError(
            "PySceneDetect is required. Install project dependencies with "
            'python -m pip install -e ".[dev]".'
        ) from exc

    if duration_seconds <= 0:
        return []

    try:
        video = open_video(str(video_path))
        manager = SceneManager()
        detector_options = {
            "threshold": threshold,
            "min_scene_len": max(1, round(min_scene_seconds * video.frame_rate)),
        }
        if progress_callback is None:
            detector = ContentDetector(**detector_options)
        else:
            total_frames = int(video.duration.frame_num) if video.duration is not None else 0
            report_interval = max(1, round(video.frame_rate))
            detection_started = time.perf_counter()

            class ProgressContentDetector(ContentDetector):
                """ContentDetector that emits throttled progress without changing results."""

                def process_frame(self, *args, **kwargs):
                    cuts = super().process_frame(*args, **kwargs)
                    frame_position = args[0] if args else kwargs.get("frame_num")
                    if hasattr(frame_position, "get_frames"):
                        frames_processed = int(frame_position.get_frames()) + 1
                    else:
                        frames_processed = int(frame_position) + 1
                    if frames_processed % report_interval == 0 or (
                        total_frames and frames_processed >= total_frames
                    ):
                        elapsed = time.perf_counter() - detection_started
                        frames_per_second = frames_processed / elapsed if elapsed else 0.0
                        remaining = (
                            max(total_frames - frames_processed, 0) / frames_per_second
                            if frames_per_second and total_frames
                            else None
                        )
                        progress_callback(
                            {
                                "frames_processed": min(frames_processed, total_frames)
                                if total_frames
                                else frames_processed,
                                "total_frames": total_frames,
                                "elapsed_seconds": round(elapsed, 1),
                                "frames_per_second": round(frames_per_second, 1),
                                "estimated_remaining_seconds": (
                                    round(remaining, 1) if remaining is not None else None
                                ),
                            }
                        )
                    return cuts

            detector = ProgressContentDetector(**detector_options)

        manager.add_detector(detector)
        manager.detect_scenes(video=video, show_progress=show_progress)
        detected = manager.get_scene_list(start_in_scene=True)
    except Exception as exc:
        raise MediaToolError(f"Scene detection failed for {video_path}: {exc}") from exc

    intervals = [
        (max(0.0, start.get_seconds()), min(duration_seconds, end.get_seconds()))
        for start, end in detected
        if end.get_seconds() > start.get_seconds()
    ]
    intervals = [(start, end) for start, end in intervals if end - start >= 0.05]
    if not intervals:
        return [(0.0, duration_seconds)]
    if intervals[0][0] > 0.05:
        intervals.insert(0, (0.0, intervals[0][0]))
    if duration_seconds - intervals[-1][1] > 0.05:
        intervals.append((intervals[-1][1], duration_seconds))
    return intervals
