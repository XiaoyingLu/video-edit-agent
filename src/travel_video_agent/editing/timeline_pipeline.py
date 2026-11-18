"""Build and review an ordered shot timeline without rendering media."""

import json
from pathlib import Path

from ..artifact_tracking import register_artifact, validate_artifact
from ..music.schemas import MusicAnalysis
from .draft_schemas import ProjectDraftSelection
from .intent_schemas import EditingPlan
from .timeline_schemas import DraftTimelineClip, ProjectDraftTimeline

_PACING_RANGES = {
    "slow": (4.0, 8.0),
    "moderate": (2.5, 5.0),
    "fast": (1.0, 3.5),
}

_ROLE_PACING_RANGES = {
    "slow": {
        "establishing": (5.0, 8.0),
        "scenic_breath": (5.0, 8.0),
        "action": (3.0, 5.0),
        "detail": (3.0, 5.0),
        "reaction": (3.0, 5.0),
        "portrait": (4.0, 6.0),
        "transition": (2.0, 4.0),
    },
    "moderate": {
        "establishing": (3.5, 5.5),
        "scenic_breath": (3.5, 5.5),
        "action": (2.0, 4.0),
        "detail": (2.0, 3.5),
        "reaction": (2.5, 4.0),
        "portrait": (2.5, 4.0),
        "transition": (1.5, 3.0),
    },
    "fast": {
        "establishing": (2.5, 4.0),
        "scenic_breath": (2.5, 4.0),
        "action": (1.0, 2.5),
        "detail": (1.0, 2.0),
        "reaction": (1.5, 2.5),
        "portrait": (1.5, 2.5),
        "transition": (1.0, 2.0),
    },
}


def _pacing_guidance(
    duration: float,
    pacing: str,
    visual_role: str = "unknown",
    energy_multiplier: float = 1.0,
) -> tuple[str, str, float]:
    """Recommend a role- and music-aware shot duration without changing source intervals."""
    lower, upper = _ROLE_PACING_RANGES[pacing].get(visual_role, _PACING_RANGES[pacing])
    lower *= energy_multiplier
    upper *= energy_multiplier
    target = round((lower + upper) / 2, 2)
    if duration < lower:
        fit = "short"
        note = (
            f"{visual_role.replace('_', ' ').title()} shot is {duration:.1f}s; "
            f"the {pacing} pacing target is about {target:.1f}s "
            f"({lower:g}–{upper:g}s). Extend only if the source moment supports it."
        )
    elif duration > upper:
        fit = "long"
        note = (
            f"{visual_role.replace('_', ' ').title()} shot is {duration:.1f}s; "
            f"the {pacing} pacing target is about {target:.1f}s "
            f"({lower:g}–{upper:g}s). Consider trimming during review."
        )
    else:
        fit = "within"
        note = (
            f"{visual_role.replace('_', ' ').title()} shot fits the {pacing} pacing "
            f"range ({lower:g}–{upper:g}s; target about {target:.1f}s)."
        )
    return fit, note, target


def _has_cinematic_metadata(clips: list) -> bool:
    return any(clip.visual_role != "unknown" or clip.shot_scale != "unknown" for clip in clips)


def _cinematic_order(clips: list) -> list:
    """Build a deterministic visual arc using editorial roles and shot-scale contrast."""
    if len(clips) < 2 or not _has_cinematic_metadata(clips):
        return clips

    remaining = list(clips)
    opening = max(
        remaining,
        key=lambda clip: (
            clip.combined_score * 0.35
            + (30 if clip.visual_role == "establishing" else 0)
            + (18 if clip.shot_scale == "aerial" else 0)
            + (14 if clip.shot_scale in {"wide", "extreme_wide"} else 0),
            clip.video_path,
            -clip.source_start_seconds,
        ),
    )
    ordered = [opening]
    remaining.remove(opening)

    while remaining:
        previous = ordered[-1]
        is_closing = len(ordered) == len(clips) - 1

        def sequence_score(candidate, previous=previous, is_closing=is_closing):
            score = candidate.combined_score * 0.35
            role = candidate.visual_role
            previous_role = previous.visual_role

            if is_closing:
                score += {
                    "scenic_breath": 28,
                    "establishing": 20,
                    "portrait": 12,
                    "transition": -8,
                }.get(role, 0)
            elif previous_role == "establishing":
                score += {
                    "action": 24,
                    "portrait": 14,
                    "detail": 10,
                    "scenic_breath": 6,
                }.get(role, 0)
            elif previous_role == "action":
                score += {
                    "detail": 18,
                    "reaction": 16,
                    "scenic_breath": 8,
                    "establishing": 6,
                }.get(role, 0)
            elif previous_role == "detail":
                score += {
                    "action": 18,
                    "establishing": 12,
                    "scenic_breath": 8,
                }.get(role, 0)
            elif previous_role == "scenic_breath":
                score += {"action": 18, "detail": 10, "portrait": 8}.get(role, 0)
            else:
                score += {
                    "establishing": 10,
                    "action": 12,
                    "detail": 8,
                    "scenic_breath": 6,
                }.get(role, 0)

            if role == previous_role and role != "unknown":
                score -= 12
            if candidate.shot_scale == previous.shot_scale and candidate.shot_scale != "unknown":
                score -= 8
            elif {candidate.shot_scale, previous.shot_scale} & {
                "wide",
                "extreme_wide",
                "aerial",
            } and {candidate.shot_scale, previous.shot_scale} & {"close_up", "extreme_close_up"}:
                score += 7
            return (
                score,
                candidate.combined_score,
                candidate.video_path,
                -candidate.source_start_seconds,
            )

        next_clip = max(remaining, key=sequence_score)
        ordered.append(next_clip)
        remaining.remove(next_clip)
    return ordered


def _sequence_reason(clip, previous, index: int, count: int) -> str:
    if index == 1:
        if clip.visual_role == "establishing" or clip.shot_scale in {
            "wide",
            "extreme_wide",
            "aerial",
        }:
            return "Opens with a wide establishing view to introduce the destination."
        return "Opens with the strongest available shot; no clear establishing shot was identified."
    if index == count and clip.visual_role in {"scenic_breath", "establishing"}:
        return "Closes on a scenic breathing shot to give the sequence a visual landing."
    if previous and clip.shot_scale != "unknown" and clip.shot_scale != previous.shot_scale:
        return (
            f"Adds visual-scale contrast after a {previous.shot_scale.replace('_', ' ')} "
            f"shot with a {clip.shot_scale.replace('_', ' ')} composition."
        )
    if clip.visual_role != "unknown":
        return f"Adds a {clip.visual_role.replace('_', ' ')} beat to vary the visual story."
    return "Retains ranked footage order because scene-role metadata is uncertain."


def build_draft_timeline(
    folder: Path,
    selection_path: Path | None = None,
    plan_path: Path | None = None,
    music_analysis_path: Path | None = None,
) -> ProjectDraftTimeline:
    """Convert selected footage into an ordered, review-only shot timeline.

    Clip source intervals are preserved exactly. Ordering, transitions, and pacing
    are deterministic recommendations; this function never edits source files or
    writes the renderer's timeline.json.
    """
    root = folder.expanduser().resolve()
    artifact_dir = root / ".video-agent"
    selection_file = (
        (selection_path or artifact_dir / "draft-selection.json").expanduser().resolve()
    )
    plan_file = (plan_path or artifact_dir / "editing-plan.json").expanduser().resolve()
    if not selection_file.is_file():
        raise FileNotFoundError(f"Missing {selection_file}; run 'video-agent draft-edit' first.")
    if not plan_file.is_file():
        raise FileNotFoundError(f"Missing {plan_file}; run 'video-agent plan' first.")

    validate_artifact(root, selection_file)
    validate_artifact(root, plan_file)
    selection = ProjectDraftSelection.model_validate_json(
        selection_file.read_text(encoding="utf-8")
    )
    plan = EditingPlan.model_validate_json(plan_file.read_text(encoding="utf-8"))
    music_file = (
        (music_analysis_path or artifact_dir / "music-analysis.json").expanduser().resolve()
        if music_analysis_path is not None
        else None
    )
    music_analysis = None
    if music_file is not None:
        if not music_file.is_file():
            raise FileNotFoundError(f"Music analysis not found: {music_file}")
        if music_file in {selection_file, plan_file}:
            raise ValueError("music analysis cannot overwrite or alias another input artifact")
        validate_artifact(root, music_file)
        music_analysis = MusicAnalysis.model_validate_json(music_file.read_text(encoding="utf-8"))
        if Path(music_analysis.project_path).resolve() != root:
            raise ValueError("music analysis belongs to a different project folder")
        try:
            music_file.relative_to(root)
        except ValueError as exc:
            raise ValueError("music analysis must be inside the project folder") from exc
        soundtrack = Path(music_analysis.music_path)
        if not soundtrack.is_absolute():
            soundtrack = root / soundtrack
        if not soundtrack.is_file():
            raise FileNotFoundError(
                f"Soundtrack referenced by music analysis is missing: {soundtrack}"
            )
    if Path(selection.project_path).resolve() != root:
        raise ValueError("draft selection belongs to a different project folder")

    selected = [clip for clip in selection.clips if clip.selected]
    if "chronological_story" in plan.priorities or plan.style in {"vlog", "documentary"}:
        selected.sort(key=lambda clip: (clip.video_path, clip.source_start_seconds))
    else:
        # Start from the relevance ranking, then apply a metadata-aware visual arc.
        selected.sort(
            key=lambda clip: (
                -clip.combined_score,
                clip.video_path,
                clip.source_start_seconds,
            )
        )
        if plan.style in {"cinematic", "social_media"}:
            selected = _cinematic_order(selected)

    timeline_clips: list[DraftTimelineClip] = []
    previous_duration: float | None = None
    elapsed_seconds = 0.0
    for index, clip in enumerate(selected, start=1):
        if index == 1 or plan.transitions.default == "cut":
            transition = "cut"
            transition_duration = 0.0
        else:
            transition = plan.transitions.default
            transition_duration = min(
                plan.transitions.duration_seconds,
                clip.duration_seconds * 0.25,
                previous_duration * 0.25 if previous_duration is not None else 0.0,
            )
            if transition_duration <= 0:
                transition = "cut"
                transition_duration = 0.0

        pacing_fit, pacing_note, pacing_target = _pacing_guidance(
            clip.duration_seconds, plan.pacing, clip.visual_role
        )
        music_energy = "unknown"
        music_section_start = None
        music_section_end = None
        music_pacing_note = "No soundtrack energy guidance was supplied."
        if music_analysis is not None:
            timeline_start = max(0.0, elapsed_seconds - transition_duration)
            timeline_end = timeline_start + clip.duration_seconds
            section_overlaps = [
                (
                    max(
                        0.0,
                        min(timeline_end, item.end_seconds)
                        - max(timeline_start, item.start_seconds),
                    ),
                    item,
                )
                for item in music_analysis.sections
            ]
            overlap, section = max(section_overlaps, key=lambda item: item[0], default=(0.0, None))
            if section is not None and overlap > 0:
                music_energy = section.energy
                music_section_start = section.start_seconds
                music_section_end = section.end_seconds
                multiplier = {"low": 1.15, "medium": 1.0, "high": 0.8}[section.energy]
                pacing_fit, pacing_note, pacing_target = _pacing_guidance(
                    clip.duration_seconds, plan.pacing, clip.visual_role, multiplier
                )
                music_pacing_note = {
                    "low": (
                        "Quieter music section: consider allowing this shot more breathing room."
                    ),
                    "medium": (
                        "Moderate-energy music section: retain the role-aware pacing target."
                    ),
                    "high": (
                        "Higher-energy music section: consider a tighter shot if the footage "
                        "supports it."
                    ),
                }[section.energy]
            else:
                music_pacing_note = (
                    "No analyzed energy section covers this point in the soundtrack."
                )
        previous_clip = timeline_clips[-1] if timeline_clips else None
        timeline_clips.append(
            DraftTimelineClip(
                clip_id=clip.clip_id,
                sequence_index=index,
                video_path=clip.video_path,
                source_start_seconds=clip.source_start_seconds,
                source_end_seconds=clip.source_end_seconds,
                duration_seconds=clip.duration_seconds,
                combined_score=clip.combined_score,
                scene_summary=clip.scene_summary,
                transcript_excerpt=clip.transcript_excerpt,
                shot_scale=clip.shot_scale,
                visual_role=clip.visual_role,
                mood=clip.mood,
                setting=clip.setting,
                primary_subject=clip.primary_subject,
                sequence_reason=_sequence_reason(clip, previous_clip, index, len(selected)),
                pacing_target_seconds=pacing_target,
                transition_in=transition,
                transition_duration_seconds=round(transition_duration, 3),
                pacing_fit=pacing_fit,
                pacing_note=pacing_note,
                music_energy=music_energy,
                music_section_start_seconds=music_section_start,
                music_section_end_seconds=music_section_end,
                music_pacing_note=music_pacing_note,
                reasons=clip.reasons,
            )
        )
        previous_duration = clip.duration_seconds
        elapsed_seconds += max(0.0, clip.duration_seconds - transition_duration)

    duration = sum(clip.duration_seconds for clip in timeline_clips) - sum(
        clip.transition_duration_seconds for clip in timeline_clips[1:]
    )
    return ProjectDraftTimeline(
        project_path=str(root),
        source_selection=(
            str(selection_file.relative_to(root))
            if selection_file.is_relative_to(root)
            else str(selection_file)
        ),
        source_editing_plan=(
            str(plan_file.relative_to(root)) if plan_file.is_relative_to(root) else str(plan_file)
        ),
        style=plan.style,
        pacing=plan.pacing,
        target_duration_seconds=plan.target_duration_seconds,
        proposed_duration_seconds=round(duration, 3),
        proposed_clip_count=len(timeline_clips),
        clips=timeline_clips,
        source_music_analysis=(
            str(music_file.relative_to(root)) if music_file is not None else None
        ),
    )


def load_draft_timeline(
    folder: Path, timeline_path: Path | None = None
) -> tuple[ProjectDraftTimeline, Path]:
    """Load a draft timeline and ensure it belongs to the requested project."""
    root = folder.expanduser().resolve()
    destination = (
        (timeline_path or root / ".video-agent" / "optimized-timeline.json").expanduser().resolve()
    )
    if not destination.is_file():
        raise FileNotFoundError(
            f"Missing {destination}; run 'video-agent optimize-timeline' first."
        )
    validate_artifact(root, destination)
    timeline = ProjectDraftTimeline.model_validate_json(destination.read_text(encoding="utf-8"))
    if Path(timeline.project_path).resolve() != root:
        raise ValueError("draft timeline belongs to a different project folder")
    return timeline, destination


def save_draft_timeline(timeline: ProjectDraftTimeline, destination: Path) -> None:
    """Persist review decisions as JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(timeline.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    if (destination.parent / "artifact-manifest.json").is_file():
        root = Path(timeline.project_path).expanduser().resolve()
        dependencies = [
            Path(value) if Path(value).is_absolute() else root / value
            for value in (
                timeline.source_selection,
                timeline.source_editing_plan,
                timeline.source_music_analysis,
            )
            if value is not None
        ]
        register_artifact(root, destination, dependencies=dependencies)


def apply_timeline_review(
    timeline: ProjectDraftTimeline,
    approve_ids: set[str],
    reject_ids: set[str],
) -> ProjectDraftTimeline:
    """Apply explicit clip decisions and mark the sequence approved only when complete."""
    if approve_ids & reject_ids:
        raise ValueError("a clip cannot be both approved and rejected")
    clips = {clip.clip_id: clip for clip in timeline.clips}
    if len(clips) != len(timeline.clips):
        raise ValueError("clip IDs must be unique in the proposed timeline")
    unknown = (approve_ids | reject_ids) - clips.keys()
    if unknown:
        raise ValueError(f"unknown clip ID(s): {', '.join(sorted(unknown))}")

    for clip in timeline.clips:
        if clip.clip_id in approve_ids:
            clip.decision = "approved"
        elif clip.clip_id in reject_ids:
            clip.decision = "rejected"

    pending = any(clip.decision == "pending" for clip in timeline.clips)
    has_approved = any(clip.decision == "approved" for clip in timeline.clips)
    timeline.timeline_approved = not pending and has_approved
    approved = [clip for clip in timeline.clips if clip.decision == "approved"]
    timeline.approved_duration_seconds = sum(clip.duration_seconds for clip in approved) - sum(
        clip.transition_duration_seconds
        for previous, clip in zip(approved, approved[1:], strict=False)
        if clip.sequence_index == previous.sequence_index + 1 and clip.transition_in != "cut"
    )
    # Revalidate after mutable review decisions are applied.
    return ProjectDraftTimeline.model_validate(timeline.model_dump())
