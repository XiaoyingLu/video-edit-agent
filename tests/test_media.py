from pathlib import Path

import pytest

from travel_video_agent.media import discover_videos


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
