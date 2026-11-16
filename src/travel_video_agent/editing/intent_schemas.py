"""Validated intent extracted from a natural-language editing request."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TransitionPreferences(BaseModel):
    """Supported transition choices; interpreted by later planning stages."""

    model_config = ConfigDict(extra="forbid")
    default: Literal["cut", "crossfade", "dissolve"] = "cut"
    duration_seconds: float = Field(default=0.5, ge=0, le=2)


class SubtitlePreferences(BaseModel):
    """Subtitle preferences without renderer-specific instructions."""

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    position: Literal["bottom", "center", "top"] = "bottom"


class AudioPreferences(BaseModel):
    """High-level audio intent."""

    model_config = ConfigDict(extra="forbid")
    preserve_original: bool = True
    background_music: bool = False


class EditingPlan(BaseModel):
    """Safe, declarative editing intent; contains no executable operations."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    source_prompt: str = Field(min_length=1, max_length=4000)
    target_duration_seconds: float = Field(default=60, gt=0, le=3600)
    style: Literal[
        "cinematic", "vlog", "documentary", "highlight_reel", "social_media"
    ] = "cinematic"
    aspect_ratio: Literal["16:9", "9:16", "1:1"] = "16:9"
    resolution: Literal["720p", "1080p", "4k"] = "1080p"
    pacing: Literal["slow", "moderate", "fast"] = "moderate"
    priorities: list[
        Literal[
            "scenic_landscapes",
            "city_highlights",
            "people_and_activities",
            "food_and_culture",
            "architecture",
            "memorable_moments",
            "chronological_story",
        ]
    ] = Field(default_factory=lambda: ["memorable_moments"])
    transitions: TransitionPreferences = Field(default_factory=TransitionPreferences)
    subtitles: SubtitlePreferences = Field(default_factory=SubtitlePreferences)
    audio: AudioPreferences = Field(default_factory=AudioPreferences)
    remove_silence: bool = False
    assumptions: list[str] = Field(default_factory=list, max_length=10)
    approval_required: bool = True
    rendering_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_preferences(self) -> "EditingPlan":
        if len(self.priorities) != len(set(self.priorities)):
            raise ValueError("priorities must not contain duplicates")
        if self.transitions.default == "cut" and self.transitions.duration_seconds != 0:
            raise ValueError("cut transitions must have a duration of 0 seconds")
        return self
