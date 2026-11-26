from pathlib import Path

import scenedetect

from video_edit_agent.understanding.scene_detection import detect_scenes


class FakeVideo:
    def __init__(self, backend: str = "pyav") -> None:
        self.BACKEND_NAME = backend
        self.frame_rate = 30
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeManager:
    def add_detector(self, detector) -> None:
        self.detector = detector

    def detect_scenes(self, video, show_progress=False) -> None:
        self.video = video

    def get_scene_list(self, start_in_scene=True):
        return []


def test_scene_detection_prefers_pyav_with_auto_threading(monkeypatch, tmp_path: Path) -> None:
    video = FakeVideo()
    calls = []

    def fake_open_video(path, **kwargs):
        calls.append((path, kwargs))
        return video

    monkeypatch.setattr(scenedetect, "open_video", fake_open_video)
    monkeypatch.setattr(scenedetect, "SceneManager", FakeManager)

    assert detect_scenes(tmp_path / "clip.mp4", 8.0) == [(0.0, 8.0)]
    assert calls == [
        (
            str(tmp_path / "clip.mp4"),
            {"backend": "pyav", "threading_mode": "auto"},
        )
    ]
    assert video.closed


def test_scene_detection_falls_back_to_opencv_when_pyav_cannot_open(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    video = FakeVideo(backend="opencv")
    calls = []

    def fake_open_video(path, **kwargs):
        calls.append(kwargs)
        if kwargs.get("backend") == "pyav":
            raise RuntimeError("unsupported codec")
        return video

    monkeypatch.setattr(scenedetect, "open_video", fake_open_video)
    monkeypatch.setattr(scenedetect, "SceneManager", FakeManager)

    assert detect_scenes(tmp_path / "clip.mp4", 8.0) == [(0.0, 8.0)]
    assert calls == [
        {"backend": "pyav", "threading_mode": "auto"},
        {"backend": "opencv"},
    ]
    assert "falling back to OpenCV" in caplog.text
    assert video.closed


def test_scene_detection_reopens_with_opencv_if_pyav_silently_falls_back(
    monkeypatch, tmp_path: Path
) -> None:
    pyav_video = FakeVideo(backend="opencv")
    opencv_video = FakeVideo(backend="opencv")
    calls = []

    def fake_open_video(path, **kwargs):
        calls.append(kwargs)
        return pyav_video if len(calls) == 1 else opencv_video

    monkeypatch.setattr(scenedetect, "open_video", fake_open_video)
    monkeypatch.setattr(scenedetect, "SceneManager", FakeManager)

    assert detect_scenes(tmp_path / "clip.mp4", 8.0) == [(0.0, 8.0)]
    assert calls == [
        {"backend": "pyav", "threading_mode": "auto"},
        {"backend": "opencv"},
    ]
    assert pyav_video.closed
    assert opencv_video.closed
