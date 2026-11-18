"""Structured, timestamped outputs from video understanding."""

from typing import Literal

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
    shot_scale: Literal[
        "extreme_wide", "wide", "medium", "close_up", "extreme_close_up", "aerial", "unknown"
    ] = "unknown"
    camera_angle: Literal[
        "eye_level", "low_angle", "high_angle", "birds_eye", "worms_eye", "unknown"
    ] = "unknown"
    composition: Literal[
        "centered",
        "rule_of_thirds",
        "leading_lines",
        "symmetrical",
        "layered_depth",
        "negative_space",
        "unknown",
    ] = "unknown"
    lighting: Literal[
        "golden_hour",
        "blue_hour",
        "daylight",
        "overcast",
        "low_light",
        "artificial",
        "backlit",
        "mixed",
        "unknown",
    ] = "unknown"
    mood: Literal[
        "serene",
        "energetic",
        "dramatic",
        "romantic",
        "mysterious",
        "joyful",
        "contemplative",
        "tense",
        "unknown",
    ] = "unknown"
    visual_role: Literal[
        "establishing",
        "action",
        "detail",
        "reaction",
        "transition",
        "portrait",
        "scenic_breath",
        "unknown",
    ] = "unknown"
    setting: str = Field(default="", max_length=100)
    primary_subject: str = Field(default="", max_length=100)


class SceneDynamics(BaseModel):
    """Temporal action and camera movement inferred from ordered scene frames."""

    model_config = ConfigDict(extra="forbid")
    temporal_summary: str = Field(default="", max_length=300)
    action_progression: list[str] = Field(default_factory=list, max_length=6)
    subject_motion: Literal[
        "approaching",
        "receding",
        "crossing",
        "stationary",
        "interacting",
        "mixed",
        "unknown",
    ] = "unknown"
    camera_motion: Literal[
        "static",
        "pan",
        "tilt",
        "zoom",
        "tracking",
        "handheld",
        "mixed",
        "unknown",
    ] = "unknown"
    confidence: float = Field(default=0, ge=0, le=1)


class VideoScene(BaseModel):
    """A detected scene with visual observations sampled across its duration."""

    model_config = ConfigDict(extra="forbid")
    scene_id: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    summary: str
    story_function: Literal[
        "establishing",
        "action",
        "detail",
        "reaction",
        "transition",
        "portrait",
        "scenic_breath",
        "unknown",
    ] = "unknown"
    dominant_mood: Literal[
        "serene",
        "energetic",
        "dramatic",
        "romantic",
        "mysterious",
        "joyful",
        "contemplative",
        "tense",
        "unknown",
    ] = "unknown"
    observations: list[FrameObservation] = Field(default_factory=list)
    dynamics: SceneDynamics | None = None


class VideoUnderstanding(BaseModel):
    """Complete understanding artifact for one project folder."""

    model_config = ConfigDict(extra="forbid")
    project_path: str
    video_count: int = Field(ge=0)
    scene_count: int = Field(ge=0)
    scenes: list[VideoScene] = Field(default_factory=list)
    model: str
