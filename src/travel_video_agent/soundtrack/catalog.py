"""Provider-neutral catalog search contracts."""

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from .schemas import SoundtrackCandidate


class SoundtrackSearchRequest(BaseModel):
    """Normalized search intent sent to a catalog provider."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=300)
    search_terms: list[str] = Field(default_factory=list, max_length=12)
    moods: list[str] = Field(default_factory=list, max_length=8)
    genres: list[str] = Field(default_factory=list, max_length=8)
    target_duration_seconds: float = Field(gt=0, le=3600)
    duration_tolerance_seconds: float = Field(default=30, ge=0, le=600)
    tempo_min_bpm: int | None = Field(default=None, ge=40, le=240)
    tempo_max_bpm: int | None = Field(default=None, ge=40, le=240)
    instrumental_preferred: bool = True
    limit: int = Field(default=20, ge=1, le=100)
    cursor: str | None = Field(default=None, max_length=500)

    def validate_tempo_range(self) -> "SoundtrackSearchRequest":
        if (
            self.tempo_min_bpm is not None
            and self.tempo_max_bpm is not None
            and self.tempo_min_bpm > self.tempo_max_bpm
        ):
            raise ValueError("minimum tempo must not exceed maximum tempo")
        return self


class SoundtrackSearchResponse(BaseModel):
    """A page of normalized catalog results."""

    model_config = ConfigDict(extra="forbid")

    provider_id: str = Field(min_length=1, max_length=80)
    candidates: list[SoundtrackCandidate] = Field(default_factory=list, max_length=100)
    next_cursor: str | None = Field(default=None, max_length=500)


class SoundtrackCatalogProvider(Protocol):
    """Async interface implemented by an authorized catalog integration."""

    @property
    def provider_id(self) -> str:
        """Stable provider identifier."""

    async def search(
        self, request: SoundtrackSearchRequest
    ) -> SoundtrackSearchResponse:
        """Search provider catalog and return normalized candidate metadata."""
