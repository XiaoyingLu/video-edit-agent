"""Reviewable, ordered shot timeline generated from a draft selection."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DraftTimelineClip(BaseModel):
    """One ordered source interval with explicit review and transition metadata."""

    model_config = ConfigDict(extra="forbid")

    clip_id: str
    sequence_index: int = Field(ge=1)
    video_path: str
    source_start_seconds: float = Field(ge=0)
    source_end_seconds: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    combined_score: float = Field(ge=0, le=100)
    scene_summary: str | None = None
    transcript_excerpt: str | None = None
    transition_in: Literal["cut", "crossfade", "dissolve"] = "cut"
    transition_duration_seconds: float = Field(default=0, ge=0, le=2)
    pacing_fit: Literal["short", "within", "long"] = "within"
    pacing_note: str
    reasons: list[str] = Field(default_factory=list)
    decision: Literal["pending", "approved", "rejected"] = "pending"
    review_required: bool = True

    @model_validator(mode="after")
    def validate_clip(self) -> "DraftTimelineClip":
        if self.source_end_seconds <= self.source_start_seconds:
            raise ValueError("source_end_seconds must be greater than source_start_seconds")
        if abs(self.source_end_seconds - self.source_start_seconds - self.duration_seconds) > 0.02:
            raise ValueError("duration_seconds must match source interval")
        if self.transition_in == "cut" and self.transition_duration_seconds != 0:
            raise ValueError("cut transitions must have a duration of 0 seconds")
        if self.transition_in != "cut" and self.transition_duration_seconds <= 0:
            raise ValueError("non-cut transitions must have a positive duration")
        if self.transition_duration_seconds > self.duration_seconds * 0.25 + 0.001:
            raise ValueError("transition duration cannot exceed 25% of the incoming clip")
        return self


class ProjectDraftTimeline(BaseModel):
    """Ordered edit proposal; intentionally distinct from the renderer timeline."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    project_path: str
    source_selection: str = ".video-agent/draft-selection.json"
    source_editing_plan: str = ".video-agent/editing-plan.json"
    style: Literal["cinematic", "vlog", "documentary", "highlight_reel", "social_media"]
    pacing: Literal["slow", "moderate", "fast"]
    target_duration_seconds: float = Field(gt=0)
    proposed_duration_seconds: float = Field(ge=0)
    approved_duration_seconds: float = Field(default=0, ge=0)
    proposed_clip_count: int = Field(ge=0)
    clips: list[DraftTimelineClip] = Field(default_factory=list)
    approval_required: Literal[True] = True
    timeline_approved: bool = False
    rendering_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_timeline(self) -> "ProjectDraftTimeline":
        indices = [clip.sequence_index for clip in self.clips]
        if indices != list(range(1, len(self.clips) + 1)):
            raise ValueError("sequence_index values must be contiguous and ordered from 1")
        clip_ids = [clip.clip_id for clip in self.clips]
        if len(clip_ids) != len(set(clip_ids)):
            raise ValueError("clip IDs must be unique in the proposed timeline")
        if self.proposed_clip_count != len(self.clips):
            raise ValueError("proposed_clip_count must match the clips list")
        total = sum(clip.duration_seconds for clip in self.clips) - sum(
            clip.transition_duration_seconds for clip in self.clips[1:]
        )
        if abs(total - self.proposed_duration_seconds) > 0.02:
            raise ValueError("proposed_duration_seconds must include transition overlaps")
        approved = [clip for clip in self.clips if clip.decision == "approved"]
        approved_duration = sum(clip.duration_seconds for clip in approved) - sum(
            clip.transition_duration_seconds for clip in approved[1:]
        )
        if abs(approved_duration - self.approved_duration_seconds) > 0.02:
            raise ValueError("approved_duration_seconds must match approved clips and transitions")
        if self.proposed_duration_seconds > self.target_duration_seconds + 0.02:
            raise ValueError("proposed duration cannot exceed the requested duration")
        if self.clips and (
            self.clips[0].transition_in != "cut" or self.clips[0].transition_duration_seconds != 0
        ):
            raise ValueError("the first clip must begin with a cut")
        if self.timeline_approved:
            if any(clip.decision == "pending" for clip in self.clips):
                raise ValueError("timeline cannot be approved while clips are pending")
            if not any(clip.decision == "approved" for clip in self.clips):
                raise ValueError("timeline must contain at least one approved clip")
        return self
