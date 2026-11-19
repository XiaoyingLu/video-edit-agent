"""Explainable soundtrack candidate ranking with rights as a hard gate."""

from pydantic import BaseModel, ConfigDict, Field

from .licensing import evaluate_soundtrack_rights
from .schemas import (
    SoundtrackBrief,
    SoundtrackCandidate,
    SoundtrackRightsDecision,
    SoundtrackRightsPolicy,
)


class SoundtrackCandidateRanking(BaseModel):
    """Creative-fit score and independent rights decision for one candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    creative_score: float = Field(ge=0, le=1)
    score_components: dict[str, float]
    matched_tags: list[str]
    rights: SoundtrackRightsDecision


def _normalize(value: str) -> str:
    return " ".join(value.casefold().replace("-", " ").replace("_", " ").split())


def _tag_overlap(expected: list[str], actual: list[str]) -> tuple[float | None, list[str]]:
    expected_terms = {_normalize(item) for item in expected if item.strip()}
    actual_terms = {_normalize(item) for item in actual if item.strip()}
    if not expected_terms:
        return None, []
    matched = sorted(
        tag
        for tag in actual_terms
        if any(tag == term or tag in term or term in tag for term in expected_terms)
    )
    return len(matched) / len(expected_terms), matched


def rank_soundtrack_candidates(
    brief: SoundtrackBrief,
    candidates: list[SoundtrackCandidate],
    policy: SoundtrackRightsPolicy | None = None,
) -> list[SoundtrackCandidateRanking]:
    """Rank candidates deterministically; rights eligibility always takes precedence."""
    weights = {
        "mood": 0.25,
        "genre": 0.25,
        "instrumentation": 0.15,
        "duration": 0.20,
        "tempo": 0.10,
        "instrumental": 0.05,
    }
    rankings: list[SoundtrackCandidateRanking] = []

    for candidate in candidates:
        components: dict[str, float] = {}
        matched_tags: set[str] = set()
        for name, expected in (
            ("mood", brief.mood),
            ("genre", brief.genre_directions),
            ("instrumentation", brief.instrumentation),
        ):
            score, matches = _tag_overlap(expected, candidate.tags)
            if score is not None:
                components[name] = score
                matched_tags.update(matches)

        duration_scale = max(brief.target_duration_seconds * 0.5, 30.0)
        components["duration"] = max(
            0.0,
            1.0 - abs(candidate.duration_seconds - brief.target_duration_seconds) / duration_scale,
        )

        if candidate.tempo_bpm is not None:
            midpoint = (brief.tempo_min_bpm + brief.tempo_max_bpm) / 2
            half_range = max((brief.tempo_max_bpm - brief.tempo_min_bpm) / 2, 1.0)
            components["tempo"] = max(0.0, 1.0 - abs(candidate.tempo_bpm - midpoint) / half_range)

        if brief.vocals == "instrumental_preferred" and candidate.instrumental is not None:
            components["instrumental"] = 1.0 if candidate.instrumental else 0.0

        active_weight = sum(weights[name] for name in components)
        creative_score = (
            sum(weights[name] * score for name, score in components.items()) / active_weight
            if active_weight
            else 0.0
        )
        rights = evaluate_soundtrack_rights(candidate, policy or brief.rights_policy)
        rankings.append(
            SoundtrackCandidateRanking(
                candidate_id=candidate.candidate_id,
                creative_score=round(creative_score, 4),
                score_components={key: round(value, 4) for key, value in components.items()},
                matched_tags=sorted(matched_tags),
                rights=rights,
            )
        )

    status_order: dict[str, int] = {"eligible": 0, "review_required": 1, "ineligible": 2}
    return sorted(
        rankings,
        key=lambda item: (
            status_order[item.rights.status],
            -item.creative_score,
            item.candidate_id,
        ),
    )
