"""Validated transcript and speech-timeline artifacts."""

from pydantic import BaseModel, ConfigDict, Field


class SpeechSegment(BaseModel):
    """A timestamped speech segment returned by the transcription model."""

    model_config = ConfigDict(extra="forbid")
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    text: str
    confidence: float = Field(default=0.5, ge=0, le=1)


class PauseInterval(BaseModel):
    """A gap between detected speech segments; never an automatic cut instruction."""

    model_config = ConfigDict(extra="forbid")
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)


class AudioTranscript(BaseModel):
    """Speech transcript and pause metadata for one source video."""

    model_config = ConfigDict(extra="forbid")
    path: str
    duration_seconds: float = Field(ge=0)
    has_audio: bool
    language: str | None = None
    language_probability: float | None = Field(default=None, ge=0, le=1)
    segments: list[SpeechSegment] = Field(default_factory=list)
    pauses: list[PauseInterval] = Field(default_factory=list)


class ProjectTranscription(BaseModel):
    """Complete transcription artifact for a project folder."""

    model_config = ConfigDict(extra="forbid")
    project_path: str
    video_count: int = Field(ge=0)
    transcribed_count: int = Field(ge=0)
    model: str
    transcripts: list[AudioTranscript] = Field(default_factory=list)
