"""Structured, timestamped outputs from video understanding."""

from pydantic import BaseModel, ConfigDict, Field


class FrameObservation(BaseModel):
    """What a vision model observes in one representative frame."""

    model_config = ConfigDict(extra="forbid")
    timestamp_seconds: float = Field(ge=0)
    frame_path: str
    description: str
    objects: list[str] = Field(default_factory=list)
    activities: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)


class VideoScene(BaseModel):
    """A detected scene with visual observations sampled across its duration."""

    model_config = ConfigDict(extra="forbid")
    scene_id: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    summary: str
    observations: list[FrameObservation] = Field(default_factory=list)


class VideoUnderstanding(BaseModel):
    """Complete understanding artifact for one project folder."""

    model_config = ConfigDict(extra="forbid")
    project_path: str
    video_count: int = Field(ge=0)
    scene_count: int = Field(ge=0)
    scenes: list[VideoScene] = Field(default_factory=list)
    model: str
