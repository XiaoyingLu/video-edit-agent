"""Content-aware soundtrack planning and discovery."""

from .brief import create_soundtrack_brief
from .catalog import (
    SoundtrackCatalogProvider,
    SoundtrackSearchRequest,
    SoundtrackSearchResponse,
)
from .licensing import evaluate_soundtrack_rights
from .ranking import SoundtrackCandidateRanking, rank_soundtrack_candidates
from .schemas import (
    SoundtrackBrief,
    SoundtrackCandidate,
    SoundtrackEnergySegment,
    SoundtrackLicense,
    SoundtrackLicenseType,
    SoundtrackRightsDecision,
    SoundtrackRightsPolicy,
)

__all__ = [
    "SoundtrackBrief",
    "SoundtrackCandidate",
    "SoundtrackCandidateRanking",
    "SoundtrackCatalogProvider",
    "SoundtrackEnergySegment",
    "SoundtrackLicense",
    "SoundtrackLicenseType",
    "SoundtrackRightsDecision",
    "SoundtrackRightsPolicy",
    "SoundtrackSearchRequest",
    "SoundtrackSearchResponse",
    "create_soundtrack_brief",
    "evaluate_soundtrack_rights",
    "rank_soundtrack_candidates",
]
