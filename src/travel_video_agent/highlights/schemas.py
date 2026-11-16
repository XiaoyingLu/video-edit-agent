"""Structured highlight candidates and explainable visual signals."""

from pydantic import BaseModel, ConfigDict, Field


class HighlightSignals(BaseModel):
    """Normalized visual measurements for one scene."""

    model_config = ConfigDict(extra="forbid")
    motion: float = Field(ge=0, le=1)
    sharpness: float = Field(ge=0, le=1)
    exposure: float = Field(ge=0, le=1)
    saturation: float = Field(ge=0, le=1)


class HighlightCandidate(BaseModel):
    """A scene suggested as a potential highlight; never an automatic edit."""

    model_config = ConfigDict(extra="forbid")
    candidate_id: str
    video_path: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    score: float = Field(ge=0, le=100)
    signals: HighlightSignals
    reasons: list[str] = Field(default_factory=list)
    scene_summary: str | None = None
    review_required: bool = True


class ProjectHighlights(BaseModel):
    """All explainable highlight candidates discovered in a project."""

    model_config = ConfigDict(extra="forbid")
    project_path: str
    video_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    candidates: list[HighlightCandidate] = Field(default_factory=list)
    scoring_version: str = "visual-v1"
