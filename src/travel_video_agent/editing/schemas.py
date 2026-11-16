"""Validated, review-first edit proposal artifacts."""

from pydantic import BaseModel, ConfigDict, Field


class EditReviewItem(BaseModel):
    """A candidate interval for a human to inspect; never an executable cut."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    suggested_action: str = "consider_removing"
    decision: str = "pending"
    review_required: bool = True
    review_reason: str
    scene_id: str | None = None
    scene_summary: str | None = None


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
