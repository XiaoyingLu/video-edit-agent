"""Local face detection used to position per-shot social export crops."""

from pathlib import Path

import numpy as np


def detect_face_crop_anchors(
    clips: list[tuple[Path, float, float, bool, str, float]],
) -> list[tuple[float, float]]:
    """Return one normalized face-center anchor per clip, falling back to frame center."""
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

    anchors: list[tuple[float, float]] = []
    for source, start, end, *_ in clips:
        capture = cv2.VideoCapture(str(source))
        centers: list[tuple[float, float]] = []
        try:
            if not capture.isOpened():
                anchors.append((0.5, 0.5))
                continue
            for fraction in (0.25, 0.5, 0.75):
                timestamp = start + (end - start) * fraction
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                success, frame = capture.read()
                if not success or frame is None:
                    continue
                height, width = frame.shape[:2]
                if width <= 0 or height <= 0:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(32, 32),
                )
                if len(faces) == 0:
                    continue
                x, y, face_width, face_height = max(
                    faces, key=lambda item: int(item[2]) * int(item[3])
                )
                centers.append(
                    (
                        min(1.0, max(0.0, (float(x) + float(face_width) / 2) / width)),
                        min(
                            1.0,
                            max(0.0, (float(y) + float(face_height) / 2) / height),
                        ),
                    )
                )
        finally:
            capture.release()

        if centers:
            values = np.asarray(centers, dtype=float)
            anchors.append(
                (
                    float(np.median(values[:, 0])),
                    float(np.median(values[:, 1])),
                )
            )
        else:
            anchors.append((0.5, 0.5))
    return anchors
