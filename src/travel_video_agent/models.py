"""Validated data models for media analysis."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class MediaInfo(BaseModel):
    """Metadata extracted from one source media file."""

    model_config = ConfigDict(extra="forbid")

    path: str
    duration_seconds: float = Field(ge=0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    frame_rate: float | None = Field(default=None, gt=0)
    video_codec: str | None = None
    audio_codec: str | None = None
    has_video: bool
    has_audio: bool


class ProjectAnalysis(BaseModel):
    """Serializable analysis summary for a media folder."""

    model_config = ConfigDict(extra="forbid")

    project_path: str
    video_count: int = Field(ge=0)
    total_duration_seconds: float = Field(ge=0)
    videos: list[MediaInfo] = Field(default_factory=list)


def relative_media_path(path: Path, root: Path) -> str:
    """Return a stable project-relative path where possible."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name
