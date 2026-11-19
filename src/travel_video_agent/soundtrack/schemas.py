"""Validated soundtrack requirements and rights policy."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SoundtrackEnergySegment(BaseModel):
    """Desired soundtrack energy for one narrative section."""

    model_config = ConfigDict(extra="forbid")

    scene_id: str = Field(min_length=1, max_length=80)
    scene_title: str = Field(min_length=1, max_length=120)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    energy: Literal["low", "medium", "high"]
    emotional_intent: str = Field(default="", max_length=200)


class SoundtrackRightsPolicy(BaseModel):
    """Minimum rights required before a candidate can be recommended."""

    model_config = ConfigDict(extra="forbid")

    commercial_use_required: bool = True
    monetized_video_allowed_required: bool = True
    attribution_acceptable: bool = True
    derivatives_and_video_sync_required: bool = True
    exclude_unclear_licenses: bool = True


class SoundtrackBrief(BaseModel):
    """Content-derived musical direction; not a license or a selected track."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    project_path: str
    source_video_plan: str
    target_duration_seconds: float = Field(gt=0, le=3600)
    video_style: Literal[
        "cinematic", "vlog", "documentary", "highlight_reel", "social_media"
    ]
    narrative_summary: str = Field(min_length=1, max_length=600)
    mood: list[str] = Field(min_length=1, max_length=8)
    genre_directions: list[str] = Field(min_length=1, max_length=6)
    instrumentation: list[str] = Field(min_length=1, max_length=8)
    vocals: Literal["instrumental_preferred", "optional", "vocals_welcome"]
    tempo_min_bpm: int = Field(ge=40, le=240)
    tempo_max_bpm: int = Field(ge=40, le=240)
    energy_arc: list[SoundtrackEnergySegment] = Field(min_length=1)
    search_terms: list[str] = Field(min_length=1, max_length=12)
    generation_prompt: str = Field(min_length=1, max_length=1200)
    rights_policy: SoundtrackRightsPolicy = Field(default_factory=SoundtrackRightsPolicy)

    @model_validator(mode="after")
    def validate_brief(self) -> "SoundtrackBrief":
        if self.tempo_min_bpm > self.tempo_max_bpm:
            raise ValueError("minimum tempo must not exceed maximum tempo")
        if abs(self.energy_arc[-1].end_seconds - self.target_duration_seconds) > 0.05:
            raise ValueError("soundtrack energy arc must cover the target duration")
        previous_end = 0.0
        for segment in self.energy_arc:
            if abs(segment.start_seconds - previous_end) > 0.05:
                raise ValueError("soundtrack energy segments must be contiguous and ordered")
            if segment.end_seconds <= segment.start_seconds:
                raise ValueError("soundtrack energy segment end must exceed its start")
            previous_end = segment.end_seconds
        return self
