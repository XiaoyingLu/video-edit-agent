"""Validated soundtrack requirements and rights policy."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class SoundtrackEnergySegment(BaseModel):
    """Desired soundtrack energy for one narrative section."""

    model_config = ConfigDict(extra="forbid")

    scene_id: str = Field(min_length=1, max_length=80)
    scene_title: str = Field(min_length=1, max_length=120)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    energy: Literal["low", "medium", "high"]
    emotional_intent: str = Field(default="", max_length=200)
    mood: str = Field(default="", max_length=80)


class SoundtrackRightsPolicy(BaseModel):
    """Minimum rights required before a candidate can be recommended."""

    model_config = ConfigDict(extra="forbid")

    commercial_use_required: bool = True
    monetized_video_allowed_required: bool = True
    attribution_acceptable: bool = True
    derivatives_and_video_sync_required: bool = True
    exclude_unclear_licenses: bool = True


class SoundtrackLicenseType(StrEnum):
    """Recognized license families; custom terms still require explicit evidence."""

    PUBLIC_DOMAIN = "public_domain"
    CC0 = "cc0"
    CC_BY = "cc_by"
    CC_BY_SA = "cc_by_sa"
    CC_BY_NC = "cc_by_nc"
    CC_BY_ND = "cc_by_nd"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


class SoundtrackLicense(BaseModel):
    """License claims and evidence supplied by a catalog or generation provider."""

    model_config = ConfigDict(extra="forbid")

    license_type: SoundtrackLicenseType = SoundtrackLicenseType.UNKNOWN
    license_name: str | None = Field(default=None, max_length=200)
    license_url: HttpUrl | None = None
    license_version: str | None = Field(default=None, max_length=80)
    evidence_url: HttpUrl | None = None
    commercial_use_allowed: bool | None = None
    monetized_video_allowed: bool | None = None
    derivatives_allowed: bool | None = None
    synchronization_allowed: bool | None = None
    attribution_required: bool = False
    attribution_text: str | None = Field(default=None, max_length=1000)
    terms_snapshot: str | None = Field(default=None, max_length=2000)
    account_constraints: list[str] = Field(default_factory=list, max_length=20)


class SoundtrackCandidate(BaseModel):
    """Normalized audio candidate with source provenance and license evidence."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1, max_length=200)
    origin: Literal["catalog", "generated"]
    provider_id: str = Field(min_length=1, max_length=80)
    provider_asset_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=200)
    creator: str | None = Field(default=None, max_length=200)
    duration_seconds: float = Field(gt=0, le=7200)
    tempo_bpm: int | None = Field(default=None, ge=40, le=240)
    instrumental: bool | None = None
    tags: list[str] = Field(default_factory=list, max_length=40)
    source_url: HttpUrl
    preview_url: HttpUrl | None = None
    asset_path: str | None = Field(default=None, max_length=1000)
    retrieved_at: datetime
    asset_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    license: SoundtrackLicense


class SoundtrackCandidateCollection(BaseModel):
    """Project-local collection combining catalog and generated audio candidates."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    project_path: str
    brief_path: str
    catalog_candidates_path: str | None = None
    candidates: list[SoundtrackCandidate] = Field(default_factory=list, max_length=200)


class SoundtrackRightsDecision(BaseModel):
    """Explainable result of evaluating a candidate against an active policy."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    status: Literal["eligible", "ineligible", "review_required"]
    reasons: list[str] = Field(default_factory=list)
    checked_policy: SoundtrackRightsPolicy


class SoundtrackBrief(BaseModel):
    """Content-derived musical direction; not a license or a selected track."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    project_path: str
    source_video_plan: str
    target_duration_seconds: float = Field(gt=0, le=3600)
    video_style: Literal["cinematic", "vlog", "documentary", "highlight_reel", "social_media"]
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
