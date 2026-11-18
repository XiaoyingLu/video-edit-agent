"""Explicit storyboard approval and deterministic production-timeline compilation."""

import json
from pathlib import Path

from ..artifact_tracking import register_artifact, validate_artifact
from .planning_schemas import VideoProjectPlan
from .timeline_pipeline import _pacing_guidance, save_draft_timeline
from .timeline_schemas import DraftTimelineClip, ProjectDraftTimeline


def approve_storyboard(
    folder: Path, *, plan_path: Path | None = None
) -> tuple[VideoProjectPlan, Path]:
    """Explicitly approve a fully footage-mapped storyboard for timeline compilation."""
    root = folder.expanduser().resolve()
    destination = (plan_path or root / ".video-agent" / "video-project-plan.json").expanduser().resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("project plan must be inside the project folder") from exc
    if not destination.is_file():
        raise FileNotFoundError(f"Missing {destination}; create a storyboard first.")
    validate_artifact(root, destination)
    plan = VideoProjectPlan.model_validate_json(destination.read_text(encoding="utf-8"))
    if Path(plan.project_path).resolve() != root:
        raise ValueError("video project plan belongs to a different project folder")
    if plan.planning_stage not in {"footage_mapped", "production_ready"}:
        raise ValueError("storyboard approval requires every shot to have an approved footage match")
    shots = [shot for scene in plan.scenes for shot in scene.shots]
    if not shots or any(shot.assignment_status != "approved" for shot in shots):
        raise ValueError("every storyboard shot must have an approved footage assignment")
    for shot in shots:
        source = (root / shot.footage_path).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"source footage for {shot.shot_id} escapes the project folder") from exc
        if not source.is_file():
            raise ValueError(f"source footage for {shot.shot_id} does not exist: {source}")
        if shot.footage_start_seconds is None or shot.footage_end_seconds is None:
            raise ValueError(f"source interval is missing for {shot.shot_id}")
    plan.approved = True
    plan.planning_stage = "production_ready"
    destination.write_text(json.dumps(plan.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    register_artifact(root, destination)
    return plan, destination


def build_storyboard_timeline(
    folder: Path,
    *,
    plan_path: Path | None = None,
) -> ProjectDraftTimeline:
    """Compile approved storyboard shots into pending, review-only timeline clips.

    Source intervals are center-trimmed to storyboard target duration when enough
    footage is available. Shorter source intervals are preserved without stretching.
    """
    root = folder.expanduser().resolve()
    plan_file = (plan_path or root / ".video-agent" / "video-project-plan.json").expanduser().resolve()
    try:
        plan_file.relative_to(root)
    except ValueError as exc:
        raise ValueError("project plan must be inside the project folder") from exc
    if not plan_file.is_file():
        raise FileNotFoundError(f"Missing {plan_file}; generate and approve the storyboard first.")
    validate_artifact(root, plan_file)
    plan = VideoProjectPlan.model_validate_json(plan_file.read_text(encoding="utf-8"))
    if Path(plan.project_path).resolve() != root:
        raise ValueError("video project plan belongs to a different project folder")
    if plan.planning_stage != "production_ready" or not plan.approved:
        raise ValueError("approve the fully footage-mapped storyboard before compiling a timeline")

    editing = plan.editing_preferences
    clips: list[DraftTimelineClip] = []
    previous_duration: float | None = None
    for scene in plan.scenes:
        for shot in scene.shots:
            if shot.assignment_status != "approved" or not shot.footage_path:
                raise ValueError(f"shot {shot.shot_id} does not have an approved footage assignment")
            source = (root / shot.footage_path).resolve()
            try:
                source.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"source footage for {shot.shot_id} escapes the project folder") from exc
            if not source.is_file():
                raise ValueError(f"source footage for {shot.shot_id} does not exist: {source}")
            if shot.footage_start_seconds is None or shot.footage_end_seconds is None:
                raise ValueError(f"source interval is missing for {shot.shot_id}")

            available = shot.footage_end_seconds - shot.footage_start_seconds
            duration = min(available, shot.target_duration_seconds)
            offset = max(0.0, (available - duration) / 2)
            source_start = shot.footage_start_seconds + offset
            source_end = source_start + duration

            transition = "cut"
            transition_duration = 0.0
            if clips and editing.transitions.default != "cut":
                transition = editing.transitions.default
                transition_duration = min(
                    editing.transitions.duration_seconds,
                    duration * 0.25,
                    (previous_duration or 0.0) * 0.25,
                )
                if transition_duration <= 0:
                    transition = "cut"

            pacing_fit, pacing_note, pacing_target = _pacing_guidance(
                duration, editing.pacing, shot.visual_role
            )
            clips.append(
                DraftTimelineClip(
                    clip_id=shot.shot_id,
                    sequence_index=len(clips) + 1,
                    video_path=shot.footage_path,
                    source_start_seconds=round(source_start, 3),
                    source_end_seconds=round(source_end, 3),
                    duration_seconds=round(duration, 3),
                    combined_score=100.0,
                    scene_summary=f"{scene.title}: {shot.visual_description}",
                    shot_scale=shot.shot_scale,
                    visual_role=shot.visual_role,
                    sequence_reason=f"Storyboard order: {scene.title} — {shot.purpose}",
                    pacing_target_seconds=pacing_target,
                    transition_in=transition,
                    transition_duration_seconds=round(transition_duration, 3),
                    pacing_fit=pacing_fit,
                    pacing_note=(
                        pacing_note
                        if available + 0.001 >= shot.target_duration_seconds
                        else f"Assigned source contains {available:.1f}s, shorter than the "
                        f"{shot.target_duration_seconds:.1f}s storyboard target; footage was "
                        "not stretched."
                    ),
                    reasons=["Explicitly approved storyboard footage assignment."],
                )
            )
            previous_duration = duration

    proposed = sum(clip.duration_seconds for clip in clips) - sum(
        clip.transition_duration_seconds for clip in clips[1:]
    )
    return ProjectDraftTimeline(
        project_path=str(root),
        source_selection=plan_file.relative_to(root).as_posix(),
        source_editing_plan=plan.source_editing_plan,
        style=editing.style,
        pacing=editing.pacing,
        target_duration_seconds=plan.brief.target_duration_seconds,
        proposed_duration_seconds=round(proposed, 3),
        proposed_clip_count=len(clips),
        clips=clips,
    )


def save_storyboard_timeline(
    timeline: ProjectDraftTimeline, destination: Path
) -> None:
    """Persist the compiled timeline and register storyboard dependencies."""
    save_draft_timeline(timeline, destination)
    root = Path(timeline.project_path).expanduser().resolve()
    plan_file = root / timeline.source_selection
    editing_file = root / timeline.source_editing_plan
    register_artifact(root, destination, dependencies=[plan_file, editing_file])
