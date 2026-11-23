"""Local face detection and temporal crop paths for social exports."""

from pathlib import Path

import numpy as np


def _detector():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Face-aware reframing requires OpenCV; install the project's standard dependencies."
        ) from exc
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Could not load the OpenCV face detector: {cascade_path}")
    return cv2, detector


def detect_face_crop_paths(
    clips: list[tuple[Path, float, float, bool, str, float]],
    *,
    sample_interval_seconds: float = 0.5,
    max_samples_per_clip: int = 60,
) -> list[list[tuple[float, float, float]]]:
    """Track the most plausible face through each clip and return smoothed crop keyframes.

    Coordinates and timestamps are normalized to the clip's own frame and duration.
    A short moving average suppresses detector jitter. Clips with fewer than two
    detections return an empty path so the renderer can use its center fallback.
    """
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be greater than zero")
    if max_samples_per_clip < 2:
        raise ValueError("max_samples_per_clip must be at least two")
    cv2, detector = _detector()
    paths: list[list[tuple[float, float, float]]] = []

    for source, start, end, *_ in clips:
        duration = end - start
        if duration <= 0:
            paths.append([])
            continue
        interval = max(sample_interval_seconds, duration / (max_samples_per_clip - 1))
        sample_times = np.arange(0.0, duration + interval * 0.25, interval).tolist()
        if not sample_times or sample_times[0] != 0.0:
            sample_times.insert(0, 0.0)
        if sample_times[-1] < duration:
            sample_times.append(duration)
        sample_times = sample_times[:max_samples_per_clip]
        capture = cv2.VideoCapture(str(source))
        detections: list[tuple[float, float, float]] = []
        previous: tuple[float, float] | None = None
        try:
            if not capture.isOpened():
                paths.append([])
                continue
            for relative_time in sample_times:
                timestamp = min(end, start + relative_time)
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                success, frame = capture.read()
                if not success or frame is None:
                    continue
                height, width = frame.shape[:2]
                if width <= 0 or height <= 0:
                    continue
                # Detection on a half-size frame reduces CPU cost without changing
                # normalized coordinates.
                small = cv2.resize(frame, (max(1, width // 2), max(1, height // 2)))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(16, 16)
                )
                if len(faces) == 0:
                    continue
                candidates = []
                for x, y, face_width, face_height in faces:
                    center = (
                        min(1.0, max(0.0, (float(x) + float(face_width) / 2) / (width / 2))),
                        min(1.0, max(0.0, (float(y) + float(face_height) / 2) / (height / 2))),
                    )
                    area = float(face_width * face_height)
                    distance = (
                        0.0
                        if previous is None
                        else (center[0] - previous[0]) ** 2 + (center[1] - previous[1]) ** 2
                    )
                    candidates.append((center, area, distance))
                if previous is None:
                    center = max(candidates, key=lambda item: item[1])[0]
                else:
                    nearest = min(candidates, key=lambda item: item[2])
                    largest = max(candidates, key=lambda item: item[1])
                    center = nearest[0] if nearest[2] <= 0.16 else largest[0]
                previous = center
                detections.append((float(relative_time), center[0], center[1]))
        finally:
            capture.release()

        if len(detections) < 2:
            paths.append([])
            continue
        # Three-point moving average keeps motion responsive while reducing
        # frame-to-frame jumps from Haar detector bounding-box noise.
        smoothed: list[tuple[float, float, float]] = []
        for index, point in enumerate(detections):
            window = detections[max(0, index - 1) : min(len(detections), index + 2)]
            smoothed.append(
                (
                    point[0],
                    float(np.mean([item[1] for item in window])),
                    float(np.mean([item[2] for item in window])),
                )
            )
        paths.append(smoothed)
    return paths


def detect_face_crop_anchors(
    clips: list[tuple[Path, float, float, bool, str, float]],
) -> list[tuple[float, float]]:
    """Compatibility helper returning one median face center per clip."""
    paths = detect_face_crop_paths(clips)
    anchors: list[tuple[float, float]] = []
    for path in paths:
        if path:
            anchors.append(
                (
                    float(np.median([point[1] for point in path])),
                    float(np.median([point[2] for point in path])),
                )
            )
        else:
            anchors.append((0.5, 0.5))
    return anchors
