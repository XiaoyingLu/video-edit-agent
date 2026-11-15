import pytest
from pydantic import ValidationError

from travel_video_agent.models import MediaInfo, ProjectAnalysis


def test_media_info_accepts_valid_metadata() -> None:
    media = MediaInfo(
        path="clip.mp4",
        duration_seconds=12.5,
        width=1920,
        height=1080,
        frame_rate=30,
        video_codec="h264",
        audio_codec="aac",
        has_video=True,
        has_audio=True,
    )
    assert media.width == 1920


def test_media_info_rejects_negative_duration() -> None:
    with pytest.raises(ValidationError):
        MediaInfo(
            path="clip.mp4",
            duration_seconds=-1,
            has_video=True,
            has_audio=False,
        )


def test_project_analysis_totals_are_non_negative() -> None:
    project = ProjectAnalysis(
        project_path="/tmp/trip",
        video_count=0,
        total_duration_seconds=0,
    )
    assert project.videos == []
