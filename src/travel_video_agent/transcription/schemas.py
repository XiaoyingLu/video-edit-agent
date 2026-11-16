"""Validated timestamped speech transcription artifacts."""

from pydantic import BaseModel, ConfigDict, Field


class TranscriptSegment(BaseModel):
    """One recognized speech segment with source-media timestamps."""

    model_config = ConfigDict(extra="forbid")

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    text: str
    average_log_probability: float | None = None
    no_speech_probability: float | None = Field(default=None, ge=0, le=1)


class VideoTranscript(BaseModel):
    """Transcription result for one video."""

    model_config = ConfigDict(extra="forbid")

    path: str
    duration_seconds: float = Field(ge=0)
    has_audio: bool
    language: str | None = None
    language_probability: float | None = Field(default=None, ge=0, le=1)
    segments: list[TranscriptSegment] = Field(default_factory=list)


class ProjectTranscription(BaseModel):
    """Serializable transcription artifact for a project folder."""

    model_config = ConfigDict(extra="forbid")

    project_path: str
    video_count: int = Field(ge=0)
    transcribed_video_count: int = Field(ge=0)
    segment_count: int = Field(ge=0)
    model: str
    videos: list[VideoTranscript] = Field(default_factory=list)
