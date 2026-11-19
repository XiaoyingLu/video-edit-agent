"""Warnings for source-footage reuse across storyboard shots."""

import warnings
from collections.abc import Iterable


def warn_for_overlapping_footage(
    assignments: Iterable[tuple[str, str, float, float]], *, stage: str
) -> None:
    """Warn when different shots reuse overlapping intervals from the same source video."""
    items = list(assignments)
    for index, (first_id, first_path, first_start, first_end) in enumerate(items):
        for second_id, second_path, second_start, second_end in items[index + 1 :]:
            if first_path != second_path:
                continue
            overlap_start = max(first_start, second_start)
            overlap_end = min(first_end, second_end)
            if overlap_end <= overlap_start:
                continue
            warnings.warn(
                f"{stage}: shots '{first_id}' and '{second_id}' reuse overlapping source "
                f"footage from '{first_path}' ({overlap_start:.2f}–{overlap_end:.2f}s). "
                "Reuse is allowed; review whether the repetition is intentional.",
                UserWarning,
                stacklevel=2,
            )
