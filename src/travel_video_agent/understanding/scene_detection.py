"""Content-based scene boundary detection using PySceneDetect."""

from pathlib import Path

from ..media import MediaToolError


def detect_scenes(
    video_path: Path,
    duration_seconds: float,
    threshold: float = 27.0,
    min_scene_seconds: float = 1.0,
) -> list[tuple[float, float]]:
    """Return scene intervals in seconds, falling back to the whole clip."""
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
        manager.add_detector(
            ContentDetector(
                threshold=threshold,
                min_scene_len=max(1, round(min_scene_seconds * video.frame_rate)),
            )
        )
        manager.detect_scenes(video=video, show_progress=False)
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
