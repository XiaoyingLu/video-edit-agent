"""Fail-closed soundtrack license eligibility checks."""

from .schemas import (
    SoundtrackCandidate,
    SoundtrackLicenseType,
    SoundtrackRightsDecision,
    SoundtrackRightsPolicy,
)


def evaluate_soundtrack_rights(
    candidate: SoundtrackCandidate,
    policy: SoundtrackRightsPolicy | None = None,
) -> SoundtrackRightsDecision:
    """Evaluate documented rights; creative relevance cannot override this result."""
    requirements = policy or SoundtrackRightsPolicy()
    license_info = candidate.license
    reasons: list[str] = []

    if license_info.license_type in {
        SoundtrackLicenseType.UNKNOWN,
        SoundtrackLicenseType.CC_BY_NC,
        SoundtrackLicenseType.CC_BY_ND,
    }:
        reasons.append(f"License type {license_info.license_type.value} is not eligible")
    elif license_info.license_type == SoundtrackLicenseType.CC_BY_SA:
        reasons.append("Share-alike license requires explicit review before use")

    if not license_info.license_name or not license_info.license_url:
        reasons.append("License name and canonical license URL must be documented")
    if requirements.exclude_unclear_licenses and not license_info.evidence_url:
        reasons.append("License evidence URL is missing")
    if requirements.commercial_use_required and license_info.commercial_use_allowed is not True:
        reasons.append("Commercial-use permission is not explicitly confirmed")
    if (
        requirements.monetized_video_allowed_required
        and license_info.monetized_video_allowed is not True
    ):
        reasons.append("Monetized-video permission is not explicitly confirmed")
    if (
        requirements.derivatives_and_video_sync_required
        and license_info.derivatives_allowed is not True
    ):
        reasons.append("Adaptation/derivative permission is not explicitly confirmed")
    if (
        requirements.derivatives_and_video_sync_required
        and license_info.synchronization_allowed is not True
    ):
        reasons.append("Video synchronization permission is not explicitly confirmed")
    if license_info.attribution_required and not requirements.attribution_acceptable:
        reasons.append("Required attribution is not accepted by the active policy")
    if license_info.attribution_required and not license_info.attribution_text:
        reasons.append("Required attribution text is missing")

    if reasons:
        status = (
            "review_required"
            if license_info.license_type == SoundtrackLicenseType.CC_BY_SA
            else "ineligible"
        )
    else:
        status = "eligible"

    return SoundtrackRightsDecision(
        candidate_id=candidate.candidate_id,
        status=status,
        reasons=reasons,
        checked_policy=requirements,
    )
