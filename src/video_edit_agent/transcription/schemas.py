"""Validated timestamped speech transcription artifacts."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..context_signals import ContextSignal, TimedBoundary


class TranscriptSegment(BaseModel):
    """One recognized speech segment with source-media timestamps."""

    model_config = ConfigDict(extra="forbid")

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    text: str
    average_log_probability: float | None = None
    no_speech_probability: float | None = Field(default=None, ge=0, le=1)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()


class PauseInterval(BaseModel):
    """A gap between speech segments, not an instruction to remove footage."""

    model_config = ConfigDict(extra="forbid")

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)


class VideoTranscript(BaseModel):
    """Transcription result for one video."""

    model_config = ConfigDict(extra="forbid")

    path: str
    duration_seconds: float = Field(ge=0)
    has_audio: bool
    language: str | None = None
    language_probability: float | None = Field(default=None, ge=0, le=1)
    segments: list[TranscriptSegment] = Field(default_factory=list)
    pauses: list[PauseInterval] = Field(default_factory=list)


class ProjectTranscription(BaseModel):
    """Serializable transcription artifact for a project folder."""

    model_config = ConfigDict(extra="forbid")

    project_path: str
    video_count: int = Field(ge=0)
    transcribed_video_count: int = Field(ge=0)
    segment_count: int = Field(ge=0)
    model: str
    videos: list[VideoTranscript] = Field(default_factory=list)
    context_signal: ContextSignal | None = None

    @model_validator(mode="after")
    def attach_context_signal(self) -> "ProjectTranscription":
        if self.context_signal is None:
            speech = []
            silence = []
            for video in self.videos:
                speech.extend(
                    TimedBoundary(
                        video_path=video.path,
                        start_seconds=segment.start_seconds,
                        end_seconds=segment.end_seconds,
                        text=segment.text[:120],
                    )
                    for segment in video.segments
                )
                silence.extend(
                    TimedBoundary(
                        video_path=video.path,
                        start_seconds=pause.start_seconds,
                        end_seconds=pause.end_seconds,
                    )
                    for pause in video.pauses
                )
            self.context_signal = ContextSignal(
                speech_boundaries=speech[:500],
                silence_boundaries=silence[:500],
            )
        return self
