"""Deterministic, advisory pacing and transition suggestions for music-aware storyboards."""

from collections.abc import Sequence

from .planning_schemas import PlannedScene, StoryboardShot


def _pacing_for_shot(shot: StoryboardShot, overall_pacing: str) -> tuple[str, str]:
    """Recommend a qualitative shot rhythm without changing its planned duration."""
    energy = shot.music_energy
    scenic = shot.visual_role in {"scenic_breath", "establishing"} or shot.shot_scale in {
        "extreme_wide",
        "aerial",
    }
    if energy == "low":
        if scenic:
            return (
                "linger",
                "Low-energy music and a scenic or establishing image favor a longer visual breath.",
            )
        return "steady", "Low-energy music favors a measured pace without rushing the visual beat."
    if energy == "medium":
        return "steady", "Medium-energy music supports a consistent, readable shot rhythm."
    if energy == "high":
        if scenic:
            return (
                "steady",
                "High-energy music suggests momentum, while the scenic image benefits "
                "from room to read.",
            )
        return "brisk", "High-energy music supports a more concise, dynamic visual rhythm."
    fallback = {"slow": "linger", "moderate": "steady", "fast": "brisk"}.get(
        overall_pacing, "unknown"
    )
    if fallback == "unknown":
        return "unknown", "No analyzed music section or overall pacing preference is available."
    return (
        fallback,
        f"No music section covers this shot; using the requested {overall_pacing} pacing.",
    )


def _transition_for_pair(
    current: StoryboardShot,
    following: StoryboardShot | None,
) -> tuple[str, str]:
    """Suggest an editorial transition from adjacent shot properties, never a beat timestamp."""
    if following is None:
        return "end_on_hold", "Final storyboard shot; allow the ending image to resolve."
    if (
        current.composition != "unknown"
        and current.composition == following.composition
        and current.music_energy != "high"
        and following.music_energy != "high"
    ):
        return "match_cut", "Adjacent shots share a known composition, offering a visual match."
    has_motion = any(
        value in {"approaching", "receding", "crossing", "interacting", "mixed"}
        for value in (current.subject_motion, following.subject_motion)
    ) or any(
        value in {"pan", "tilt", "zoom", "tracking", "handheld", "mixed"}
        for value in (current.camera_motion, following.camera_motion)
    )
    if has_motion:
        return (
            "cut_on_action",
            "Visible subject or camera motion can carry continuity across the cut.",
        )
    energy_changes = (
        current.music_energy != "unknown"
        and following.music_energy != "unknown"
        and current.music_energy != following.music_energy
    )
    scenic_breath = (
        current.visual_role == "scenic_breath" or following.visual_role == "scenic_breath"
    )
    if energy_changes or (
        scenic_breath and current.music_energy == "low" and following.music_energy == "low"
    ):
        return (
            "soft_dissolve",
            "A musical energy change or quiet scenic passage can support a gentle transition.",
        )
    return (
        "straight_cut",
        "A clean cut preserves forward movement without implying beat synchronization.",
    )


def annotate_music_pacing(
    scenes: Sequence[PlannedScene],
    overall_pacing: str,
) -> list[PlannedScene]:
    """Attach deterministic, explainable pacing and outgoing-transition suggestions."""
    shots = [shot for scene in scenes for shot in scene.shots]
    annotated_shots: dict[str, StoryboardShot] = {}
    for index, shot in enumerate(shots):
        pacing, pacing_rationale = _pacing_for_shot(shot, overall_pacing)
        transition, transition_rationale = _transition_for_pair(
            shot,
            shots[index + 1] if index + 1 < len(shots) else None,
        )
        annotated_shots[shot.shot_id] = shot.model_copy(
            update={
                "pacing_recommendation": pacing,
                "pacing_rationale": pacing_rationale,
                "transition_recommendation": transition,
                "transition_rationale": transition_rationale,
            }
        )
    return [
        scene.model_copy(
            update={
                "shots": [annotated_shots[shot.shot_id] for shot in scene.shots],
            }
        )
        for scene in scenes
    ]
