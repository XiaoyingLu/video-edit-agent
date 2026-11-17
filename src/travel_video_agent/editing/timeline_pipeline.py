"""Build and review an ordered shot timeline without rendering media."""

import json
from pathlib import Path

from ..artifact_tracking import register_artifact, validate_artifact
from .draft_schemas import ProjectDraftSelection
from .intent_schemas import EditingPlan
from .timeline_schemas import DraftTimelineClip, ProjectDraftTimeline

_PACING_RANGES = {
    "slow": (4.0, 8.0),
    "moderate": (2.5, 5.0),
    "fast": (1.0, 3.5),
}


def _pacing_guidance(duration: float, pacing: str) -> tuple[str, str]:
    lower, upper = _PACING_RANGES[pacing]
    if duration < lower:
        return (
            "short",
            f"Short for {pacing} pacing; consider whether this shot needs more breathing room.",
        )
    if duration > upper:
        return (
            "long",
            f"Long for {pacing} pacing; consider trimming if the visual moment feels static.",
        )
    return "within", f"Within the suggested {pacing} pacing range ({lower:g}–{upper:g}s)."


def build_draft_timeline(
    folder: Path,
    selection_path: Path | None = None,
    plan_path: Path | None = None,
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
    if Path(selection.project_path).resolve() != root:
        raise ValueError("draft selection belongs to a different project folder")

    selected = [clip for clip in selection.clips if clip.selected]
    if "chronological_story" in plan.priorities or plan.style in {"vlog", "documentary"}:
        selected.sort(key=lambda clip: (clip.video_path, clip.source_start_seconds))
    else:
        # The draft selection is already ranked by its deterministic relevance score.
        selected.sort(
            key=lambda clip: (
                -clip.combined_score,
                clip.video_path,
                clip.source_start_seconds,
            )
        )

    timeline_clips: list[DraftTimelineClip] = []
    previous_duration: float | None = None
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

        pacing_fit, pacing_note = _pacing_guidance(clip.duration_seconds, plan.pacing)
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
                transition_in=transition,
                transition_duration_seconds=round(transition_duration, 3),
                pacing_fit=pacing_fit,
                pacing_note=pacing_note,
                reasons=clip.reasons,
            )
        )
        previous_duration = clip.duration_seconds

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
            for value in (timeline.source_selection, timeline.source_editing_plan)
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
