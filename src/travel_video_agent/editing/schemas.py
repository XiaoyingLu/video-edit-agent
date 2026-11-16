"""Validated, review-first edit proposal artifacts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EditReviewItem(BaseModel):
    """A candidate interval for a human to inspect; never an executable cut."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    suggested_action: Literal["consider_removing"] = "consider_removing"
    decision: Literal["pending", "approved", "rejected"] = "pending"
    review_required: bool = True
    review_reason: str
    scene_id: str | None = None
    scene_summary: str | None = None

    @model_validator(mode="after")
    def validate_interval(self) -> "EditReviewItem":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        if abs((self.end_seconds - self.start_seconds) - self.duration_seconds) > 0.01:
            raise ValueError("duration_seconds must match the interval bounds")
        return self


class VideoEditProposal(BaseModel):
    """Review items associated with one source video."""

    model_config = ConfigDict(extra="forbid")

    path: str
    duration_seconds: float = Field(ge=0)
    review_items: list[EditReviewItem] = Field(default_factory=list)


class ProjectEditProposal(BaseModel):
    """A project-level proposal generated from existing analysis artifacts."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    project_path: str
    source_artifacts: list[str]
    video_count: int = Field(ge=0)
    review_item_count: int = Field(ge=0)
    approval_required: bool = True
    rendering_performed: bool = False
    videos: list[VideoEditProposal] = Field(default_factory=list)
