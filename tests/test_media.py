from pathlib import Path

import pytest

from video_edit_agent.media import discover_videos


def test_discover_videos_is_recursive_and_case_insensitive(tmp_path: Path) -> None:
    (tmp_path / "clip.MP4").touch()
    nested = tmp_path / "day-2"
    nested.mkdir()
    (nested / "coast.mov").touch()
    (nested / "notes.txt").touch()

    found = discover_videos(tmp_path)
    assert [path.name for path in found] == ["clip.MP4", "coast.mov"]


def test_discover_videos_rejects_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_videos(tmp_path / "missing")


def test_probe_video_reads_embedded_creation_timestamp(tmp_path: Path, monkeypatch) -> None:
    import json
    from types import SimpleNamespace

    from video_edit_agent.media import probe_video

    video = tmp_path / "clip.mp4"
    video.touch()
    monkeypatch.setattr("video_edit_agent.media.shutil.which", lambda name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        "video_edit_agent.media.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout=json.dumps(
                {
                    "format": {
                        "duration": "12.5",
                        "tags": {"creation_time": "2026-06-12T09:30:00.000000Z"},
                    },
                    "streams": [
                        {
                            "codec_type": "video",
                            "codec_name": "h264",
                            "width": 1920,
                            "height": 1080,
                            "avg_frame_rate": "30/1",
                        }
                    ],
                }
            ),
            stderr="",
        ),
    )

    result = probe_video(video, tmp_path)

    assert result.capture_timestamp == "2026-06-12T09:30:00.000000Z"
    assert result.duration_seconds == 12.5
