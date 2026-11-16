"""Validated, review-first edit proposal and timeline artifacts."""

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


class TimelineSegment(BaseModel):
    """A retained source interval for a future deterministic renderer."""

    model_config = ConfigDict(extra="forbid")

    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_interval(self) -> "TimelineSegment":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("timeline segment end must be greater than start")
        return self


class VideoTimeline(BaseModel):
    """Ordered retained intervals for one source video."""

    model_config = ConfigDict(extra="forbid")

    path: str
    source_duration_seconds: float = Field(ge=0)
    retained_duration_seconds: float = Field(ge=0)
    segments: list[TimelineSegment] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_segments(self) -> "VideoTimeline":
        previous_end = 0.0
        total = 0.0
        for segment in self.segments:
            if segment.end_seconds > self.source_duration_seconds + 0.01:
                raise ValueError("timeline segment exceeds source duration")
            if segment.start_seconds < previous_end - 0.01:
                raise ValueError("timeline segments must be ordered and non-overlapping")
            previous_end = segment.end_seconds
            total += segment.end_seconds - segment.start_seconds
        if abs(total - self.retained_duration_seconds) > 0.01:
            raise ValueError("retained duration must equal the sum of timeline segments")
        return self


class ProjectTimeline(BaseModel):
    """Validated edit-decision timeline; contains no rendering instructions."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    project_path: str
    source_proposal: str = ".video-agent/edit-proposal.json"
    approved_cut_count: int = Field(ge=0)
    retained_duration_seconds: float = Field(ge=0)
    rendering_performed: bool = False
    videos: list[VideoTimeline] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_project_duration(self) -> "ProjectTimeline":
        total = sum(video.retained_duration_seconds for video in self.videos)
        if abs(total - self.retained_duration_seconds) > 0.01:
            raise ValueError("project retained duration must equal the sum of video durations")
        return self
