"""Validated audio silence observations and review-only candidates."""

from pydantic import BaseModel, ConfigDict, Field


class SilenceInterval(BaseModel):
    """An interval where the audio level is below the configured threshold."""

    model_config = ConfigDict(extra="forbid")

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)


class SilenceCandidate(BaseModel):
    """A quiet, speech-free interval that still requires human review."""

    model_config = ConfigDict(extra="forbid")

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    review_required: bool = True
    review_reason: str = (
        "Low audio level and no overlapping transcript speech; check for scenic, "
        "ambient, or intentionally quiet footage before considering an edit."
    )


class VideoSilenceAnalysis(BaseModel):
    """Audio silence analysis for one source video."""

    model_config = ConfigDict(extra="forbid")

    path: str
    duration_seconds: float = Field(ge=0)
    has_audio: bool
    transcript_available: bool = False
    audio_silences: list[SilenceInterval] = Field(default_factory=list)
    candidates: list[SilenceCandidate] = Field(default_factory=list)


class ProjectSilenceAnalysis(BaseModel):
    """Serializable project-level silence analysis report."""

    model_config = ConfigDict(extra="forbid")

    project_path: str
    video_count: int = Field(ge=0)
    analyzed_video_count: int = Field(ge=0)
    noise_threshold_db: float
    min_silence_duration: float = Field(gt=0)
    speech_padding_seconds: float = Field(ge=0)
    videos: list[VideoSilenceAnalysis] = Field(default_factory=list)
