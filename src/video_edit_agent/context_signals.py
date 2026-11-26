"""Compact cross-stage context carried by pipeline artifacts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SceneContext(BaseModel):
    """Small scene facts useful to downstream editing skills."""

    model_config = ConfigDict(extra="forbid")
    scene_id: str
    video_path: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    summary: str = Field(default="", max_length=240)
    visual_role: str = "unknown"
    mood: str = "unknown"


class TimedBoundary(BaseModel):
    """A speech or silence interval in source-media time."""

    model_config = ConfigDict(extra="forbid")
    video_path: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    text: str | None = Field(default=None, max_length=120)


class PacingConstraint(BaseModel):
    """Role-specific duration guidance in seconds."""

    model_config = ConfigDict(extra="forbid")
    role: str
    min_seconds: float = Field(gt=0)
    max_seconds: float = Field(gt=0)
    target_seconds: float = Field(gt=0)


class ContextSignal(BaseModel):
    """Versioned, compact working state passed between editing stages."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"
    scenes: list[SceneContext] = Field(default_factory=list, max_length=300)
    speech_boundaries: list[TimedBoundary] = Field(default_factory=list, max_length=500)
    silence_boundaries: list[TimedBoundary] = Field(default_factory=list, max_length=500)
    target_aspect_ratio: Literal["16:9", "9:16", "1:1"] | None = None
    pacing: Literal["slow", "moderate", "fast"] | None = None
    pacing_constraints: list[PacingConstraint] = Field(default_factory=list, max_length=8)
