"""Reviewable shot selections produced from existing project analysis."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DraftClip(BaseModel):
    """One proposed source interval in the draft sequence."""

    model_config = ConfigDict(extra="forbid")
    clip_id: str
    video_path: str
    source_start_seconds: float = Field(ge=0)
    source_end_seconds: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    highlight_score: float = Field(ge=0, le=100)
    relevance_score: float = Field(ge=0, le=100)
    combined_score: float = Field(ge=0, le=100)
    scene_summary: str | None = None
    transcript_excerpt: str | None = None
    reasons: list[str] = Field(default_factory=list)
    selected: bool = False
    decision: Literal["pending", "approved", "rejected"] = "pending"
    review_required: bool = True

    @model_validator(mode="after")
    def validate_interval(self) -> "DraftClip":
        if self.source_end_seconds <= self.source_start_seconds:
            raise ValueError("source_end_seconds must be greater than source_start_seconds")
        if abs(self.source_end_seconds - self.source_start_seconds - self.duration_seconds) > 0.02:
            raise ValueError("duration_seconds must match source interval")
        return self


class ProjectDraftSelection(BaseModel):
    """A ranked, review-only proposed sequence; not an executable render timeline."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = "1.0"
    project_path: str
    source_artifacts: list[str]
    target_duration_seconds: float = Field(gt=0)
    selected_duration_seconds: float = Field(ge=0)
    selected_clip_count: int = Field(ge=0)
    candidates_considered: int = Field(ge=0)
    clips: list[DraftClip] = Field(default_factory=list)
    approval_required: bool = True
    rendering_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_summary(self) -> "ProjectDraftSelection":
        selected = [clip for clip in self.clips if clip.selected]
        if len(selected) != self.selected_clip_count:
            raise ValueError("selected_clip_count must match selected clips")
        duration = sum(clip.duration_seconds for clip in selected)
        if abs(duration - self.selected_duration_seconds) > 0.02:
            raise ValueError("selected_duration_seconds must match selected clips")
        if self.selected_duration_seconds > self.target_duration_seconds + 0.02:
            raise ValueError("selected duration cannot exceed requested duration")
        return self
