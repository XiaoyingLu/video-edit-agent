from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from travel_video_agent.soundtrack.catalog import SoundtrackSearchRequest
from travel_video_agent.soundtrack.licensing import evaluate_soundtrack_rights
from travel_video_agent.soundtrack.schemas import (
    SoundtrackCandidate,
    SoundtrackLicense,
    SoundtrackRightsPolicy,
)


def _candidate(**license_overrides) -> SoundtrackCandidate:
    license_data = {
        "license_type": "cc0",
        "license_name": "CC0 1.0 Universal",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "evidence_url": "https://example.com/license-evidence",
        "commercial_use_allowed": True,
        "monetized_video_allowed": True,
        "derivatives_allowed": True,
        "synchronization_allowed": True,
        "attribution_required": False,
        "attribution_text": None,
        "license_version": "1.0",
        "terms_snapshot": None,
        "account_constraints": [],
    }
    license_data.update(license_overrides)
    return SoundtrackCandidate(
        candidate_id="provider:track-123",
        origin="catalog",
        provider_id="example",
        provider_asset_id="track-123",
        title="Alpine Morning",
        creator="Example Artist",
        duration_seconds=92,
        tags=["ambient", "cinematic"],
        source_url="https://example.com/tracks/track-123",
        preview_url="https://example.com/tracks/track-123/preview",
        retrieved_at=datetime.now(UTC),
        license=SoundtrackLicense(**license_data),
    )


def test_cc0_candidate_with_explicit_rights_is_eligible() -> None:
    decision = evaluate_soundtrack_rights(_candidate())

    assert decision.status == "eligible"
    assert decision.reasons == []


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"commercial_use_allowed": None}, "Commercial-use permission"),
        ({"commercial_use_allowed": False}, "Commercial-use permission"),
        ({"monetized_video_allowed": None}, "Monetized-video permission"),
        ({"synchronization_allowed": False}, "Video synchronization permission"),
        ({"derivatives_allowed": None}, "Adaptation/derivative permission"),
        ({"license_url": None}, "canonical license URL"),
        ({"evidence_url": None}, "License evidence URL"),
    ],
)
def test_missing_or_denied_rights_fail_closed(overrides, expected_reason) -> None:
    decision = evaluate_soundtrack_rights(_candidate(**overrides))

    assert decision.status == "ineligible"
    assert any(expected_reason in reason for reason in decision.reasons)


@pytest.mark.parametrize("license_type", ["cc_by_nc", "cc_by_nd", "unknown"])
def test_noncommercial_no_derivatives_and_unknown_licenses_are_rejected(
    license_type: str,
) -> None:
    decision = evaluate_soundtrack_rights(_candidate(license_type=license_type))

    assert decision.status == "ineligible"
    assert any("not eligible" in reason for reason in decision.reasons)


def test_share_alike_license_requires_explicit_review() -> None:
    decision = evaluate_soundtrack_rights(_candidate(license_type="cc_by_sa"))

    assert decision.status == "review_required"
    assert any("Share-alike" in reason for reason in decision.reasons)


def test_required_attribution_must_be_preserved() -> None:
    decision = evaluate_soundtrack_rights(
        _candidate(attribution_required=True, attribution_text=None)
    )

    assert decision.status == "ineligible"
    assert any("attribution text" in reason for reason in decision.reasons)


def test_policy_can_reject_attribution_required_tracks() -> None:
    decision = evaluate_soundtrack_rights(
        _candidate(attribution_required=True, attribution_text="Music by Example Artist"),
        SoundtrackRightsPolicy(attribution_acceptable=False),
    )

    assert decision.status == "ineligible"
    assert any("attribution is not accepted" in reason for reason in decision.reasons)


def test_search_request_rejects_invalid_tempo_range() -> None:
    with pytest.raises(ValidationError):
        SoundtrackSearchRequest(
            query="ambient cinematic",
            target_duration_seconds=90,
            tempo_min_bpm=130,
            tempo_max_bpm=90,
        )
