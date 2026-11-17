"""Validated soundtrack analysis artifacts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MusicSection(BaseModel):
    """A broad musical energy section, not a semantic music label."""

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    energy: Literal["low", "medium", "high"]
    energy_rms: float = Field(ge=0)


class MusicAnalysis(BaseModel):
    """Tempo, beat grid, and broad energy map for one soundtrack."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    project_path: str
    music_path: str
    duration_seconds: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    bpm: float | None = Field(default=None, ge=40, le=240)
    beat_confidence: float = Field(ge=0, le=1)
    beat_interval_seconds: float | None = Field(default=None, gt=0)
    beat_times_seconds: list[float] = Field(default_factory=list)
    downbeat_times_seconds: list[float] = Field(default_factory=list)
    sections: list[MusicSection] = Field(default_factory=list)
    suggested_cut_points_seconds: list[float] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
