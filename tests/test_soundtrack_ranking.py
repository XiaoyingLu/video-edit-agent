from datetime import UTC, datetime

from video_edit_agent.soundtrack.ranking import rank_soundtrack_candidates
from video_edit_agent.soundtrack.schemas import (
    SoundtrackBrief,
    SoundtrackCandidate,
    SoundtrackLicense,
)


def _brief() -> SoundtrackBrief:
    return SoundtrackBrief(
        project_path="/tmp/trip",
        source_video_plan=".video-agent/video-project-plan.json",
        target_duration_seconds=100,
        video_style="cinematic",
        narrative_summary="A calm mountain journey with a dramatic summit reveal.",
        mood=["calm", "wonder"],
        genre_directions=["ambient", "cinematic"],
        instrumentation=["piano", "strings"],
        vocals="instrumental_preferred",
        tempo_min_bpm=70,
        tempo_max_bpm=110,
        energy_arc=[
            {
                "scene_id": "scene-1",
                "scene_title": "Opening",
                "start_seconds": 0,
                "end_seconds": 100,
                "energy": "medium",
            }
        ],
        search_terms=["calm", "ambient", "instrumental"],
        generation_prompt="Original calm cinematic instrumental.",
    )


def _candidate(
    candidate_id: str,
    *,
    tags: list[str],
    duration: float = 100,
    license_type: str = "cc0",
    tempo: int | None = 90,
    instrumental: bool | None = True,
) -> SoundtrackCandidate:
    return SoundtrackCandidate(
        candidate_id=candidate_id,
        origin="catalog",
        provider_id="test",
        provider_asset_id=candidate_id,
        title=candidate_id,
        duration_seconds=duration,
        tags=tags,
        tempo_bpm=tempo,
        instrumental=instrumental,
        source_url="https://example.com/" + candidate_id,
        retrieved_at=datetime.now(UTC),
        license=SoundtrackLicense(
            license_type=license_type,
            license_name="CC0 1.0",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            evidence_url="https://example.com/license",
            commercial_use_allowed=True,
            monetized_video_allowed=True,
            derivatives_allowed=True,
            synchronization_allowed=True,
        ),
    )


def test_ranking_prefers_creative_fit_and_explains_components() -> None:
    strong = _candidate(
        "strong",
        tags=["calm", "wonder", "ambient", "cinematic", "piano", "strings"],
    )
    weak = _candidate("weak", tags=["metal", "aggressive"], duration=170)

    results = rank_soundtrack_candidates(_brief(), [weak, strong])

    assert [item.candidate_id for item in results] == ["strong", "weak"]
    assert results[0].creative_score > results[1].creative_score
    assert results[0].rights.status == "eligible"
    assert "mood" in results[0].score_components
    assert "piano" in results[0].matched_tags


def test_rights_eligibility_is_a_hard_sorting_gate() -> None:
    high_fit_but_unlicensed = _candidate(
        "unlicensed",
        tags=["calm", "wonder", "ambient", "cinematic", "piano", "strings"],
        license_type="cc_by_nc",
    )
    lower_fit_but_eligible = _candidate("eligible", tags=["ambient"])

    results = rank_soundtrack_candidates(
        _brief(), [high_fit_but_unlicensed, lower_fit_but_eligible]
    )

    assert [item.candidate_id for item in results] == ["eligible", "unlicensed"]
    assert results[0].rights.status == "eligible"
    assert results[1].rights.status == "ineligible"
    assert results[1].creative_score > results[0].creative_score


def test_optional_tempo_and_instrumental_metadata_are_not_assumed() -> None:
    candidate = _candidate(
        "metadata-light",
        tags=["ambient"],
        tempo=None,
        instrumental=None,
    )

    result = rank_soundtrack_candidates(_brief(), [candidate])[0]

    assert "tempo" not in result.score_components
    assert "instrumental" not in result.score_components


def test_ranking_is_deterministic_for_ties() -> None:
    candidates = [
        _candidate("z-track", tags=["ambient"]),
        _candidate("a-track", tags=["ambient"]),
    ]

    first = rank_soundtrack_candidates(_brief(), candidates)
    second = rank_soundtrack_candidates(_brief(), list(reversed(candidates)))

    assert [item.candidate_id for item in first] == ["a-track", "z-track"]
    assert first == second
