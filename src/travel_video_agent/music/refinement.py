"""Human-reviewed, non-destructive soundtrack timing refinement."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MusicSyncReviewDecision(BaseModel):
    """A user's decision for one suggested timeline boundary."""

    model_config = ConfigDict(extra="forbid")

    boundary_id: str = Field(min_length=1)
    status: Literal["accepted", "rejected", "skipped", "infeasible"]
    reason: str | None = None


class MusicSyncReviewResult(BaseModel):
    """Persistable provenance for one music-sync refinement review."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    source_timeline_path: str
    suggestions_path: str
    revision_path: str
    decisions: list[MusicSyncReviewDecision] = Field(default_factory=list)
    review_required: Literal[True] = True
