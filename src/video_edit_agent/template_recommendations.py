"""Explainable, tag-based recommendations for reusable editing templates."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .templates import TemplateSpec, available_template_ids, load_template


@dataclass(frozen=True)
class TemplateRecommendation:
    """A template ranked by explicit tag matches."""

    template: TemplateSpec
    source: Path
    score: int
    matches: tuple[str, ...]


def _normalize_tag(value: str) -> str:
    """Normalize user and template tags without fuzzy or semantic matching."""
    return re.sub(r"[-_\s]+", "-", value.strip().casefold())


def recommend_templates(
    *,
    moods: Iterable[str] = (),
    best_for: Iterable[str] = (),
    project_root: Path | None = None,
    limit: int = 3,
) -> list[TemplateRecommendation]:
    """Rank templates by exact normalized mood and use-case tag matches.

    Use-case matches are weighted twice as strongly as mood matches. Ties are
    resolved by template ID so the output is deterministic and explainable.
    """
    if not 1 <= limit <= 3:
        raise ValueError("recommendation limit must be between 1 and 3")

    requested = {
        "mood": {_normalize_tag(value) for value in moods if value.strip()},
        "best_for": {_normalize_tag(value) for value in best_for if value.strip()},
    }
    if not any(requested.values()):
        raise ValueError("provide at least one --mood or --best-for tag")

    recommendations: list[TemplateRecommendation] = []
    for template_id in available_template_ids(project_root):
        template, source = load_template(template_id, project_root)
        matches: list[str] = []
        score = 0
        for category, weight in (("best_for", 2), ("mood", 1)):
            available = {_normalize_tag(tag) for tag in template.tags.get(category, [])}
            for tag in sorted(requested[category] & available):
                matches.append(f"{category}:{tag}")
                score += weight
        if score:
            recommendations.append(
                TemplateRecommendation(
                    template=template,
                    source=source,
                    score=score,
                    matches=tuple(matches),
                )
            )

    recommendations.sort(key=lambda item: (-item.score, item.template.id))
    return recommendations[:limit]
