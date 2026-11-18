"""Review-only footage suggestions for creative storyboard shots."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FootageMatchCandidate(BaseModel):
    """One analyzed source scene that may satisfy a storyboard shot."""

    model_config = ConfigDict(extra="forbid")

    video_path: str
    source_scene_id: str
    source_start_seconds: float = Field(ge=0)
    source_end_seconds: float = Field(gt=0)
    score: float = Field(ge=0, le=100)
    lexical_score: float = Field(default=0, ge=0, le=100)
    semantic_similarity: float | None = Field(default=None, ge=0, le=1)
    match_method: Literal["lexical", "hybrid"] = "lexical"
    scene_summary: str
    reasons: list[str] = Field(default_factory=list)


class ShotFootageSuggestions(BaseModel):
    """Ranked footage candidates for one storyboard shot."""

    model_config = ConfigDict(extra="forbid")

    shot_id: str
    scene_id: str
    purpose: str
    visual_description: str
    target_duration_seconds: float = Field(gt=0)
    candidates: list[FootageMatchCandidate] = Field(default_factory=list)


class ProjectFootageSuggestions(BaseModel):
    """Project-wide footage matching report; it never assigns or approves footage."""

    model_config = ConfigDict(extra="forbid")

    project_path: str
    source_plan: str
    source_understanding: str
    suggestions: list[ShotFootageSuggestions] = Field(default_factory=list)
    candidate_count: int = Field(ge=0)
    matched_shot_count: int = Field(ge=0)
    embedding_model: str | None = None
    matching_method: Literal["lexical", "hybrid"] = "lexical"
    review_required: bool = True
    assignments_performed: bool = False
