"""Content-aware soundtrack planning and discovery."""

from .brief import create_soundtrack_brief
from .catalog import (
    SoundtrackCatalogProvider,
    SoundtrackSearchRequest,
    SoundtrackSearchResponse,
)
from .licensing import evaluate_soundtrack_rights
from .providers import SoundstripeAPIError, SoundstripeCatalogProvider
from .ranking import SoundtrackCandidateRanking, rank_soundtrack_candidates
from .review import (
    SoundtrackCandidateReview,
    SoundtrackReviewError,
    SoundtrackSelection,
    build_soundtrack_candidate_review,
    save_soundtrack_candidate_review,
    search_soundstripe_catalog,
    select_soundtrack_candidate,
)
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
    "SoundtrackCandidateReview",
    "SoundstripeAPIError",
    "SoundtrackReviewError",
    "SoundtrackSelection",
    "SoundstripeCatalogProvider",
    "SoundtrackCatalogProvider",
    "SoundtrackEnergySegment",
    "SoundtrackLicense",
    "SoundtrackLicenseType",
    "SoundtrackRightsDecision",
    "SoundtrackRightsPolicy",
    "SoundtrackSearchRequest",
    "SoundtrackSearchResponse",
    "build_soundtrack_candidate_review",
    "create_soundtrack_brief",
    "evaluate_soundtrack_rights",
    "rank_soundtrack_candidates",
    "save_soundtrack_candidate_review",
    "search_soundstripe_catalog",
    "select_soundtrack_candidate",
]
