"""Human-reviewed assignment of suggested source footage to storyboard shots."""

import json
from pathlib import Path

from ..artifact_tracking import register_artifact, validate_artifact
from .footage_schemas import ProjectFootageSuggestions
from .planning_schemas import VideoProjectPlan


def review_storyboard_footage(
    folder: Path,
    *,
    selections: dict[str, int] | None = None,
    rejected_shot_ids: set[str] | None = None,
    plan_path: Path | None = None,
    suggestions_path: Path | None = None,
) -> tuple[VideoProjectPlan, Path]:
    """Apply explicit 1-based candidate selections; never infer or auto-approve matches."""
    root = folder.expanduser().resolve()
    artifact_dir = root / ".video-agent"
    plan_file = (plan_path or artifact_dir / "video-project-plan.json").expanduser().resolve()
    suggestions_file = (
        suggestions_path or artifact_dir / "footage-suggestions.json"
    ).expanduser().resolve()
    for path, label in ((plan_file, "project plan"), (suggestions_file, "footage suggestions")):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{label} must be inside the project folder") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; generate it first.")
        validate_artifact(root, path)

    plan = VideoProjectPlan.model_validate_json(plan_file.read_text(encoding="utf-8"))
    suggestions = ProjectFootageSuggestions.model_validate_json(
        suggestions_file.read_text(encoding="utf-8")
    )
    if Path(plan.project_path).resolve() != root or Path(suggestions.project_path).resolve() != root:
        raise ValueError("planning artifacts belong to a different project folder")
    if plan.planning_stage not in {"storyboard_ready", "footage_mapped"}:
        raise ValueError("footage review requires a storyboard_ready or footage_mapped plan")

    selections = selections or {}
    rejected = rejected_shot_ids or set()
    if set(selections) & rejected:
        raise ValueError("a shot cannot be both selected and rejected")
    shots = {shot.shot_id: shot for scene in plan.scenes for shot in scene.shots}
    suggestion_by_id = {item.shot_id: item for item in suggestions.suggestions}
    unknown = (set(selections) | rejected) - set(shots)
    if unknown:
        raise ValueError(f"unknown storyboard shot ID(s): {', '.join(sorted(unknown))}")

    for shot_id, candidate_number in selections.items():
        if not 1 <= candidate_number <= 10:
            raise ValueError("candidate number must be between 1 and 10")
        suggestion = suggestion_by_id.get(shot_id)
        if suggestion is None or candidate_number > len(suggestion.candidates):
            raise ValueError(f"shot '{shot_id}' has no candidate #{candidate_number}")
        candidate = suggestion.candidates[candidate_number - 1]
        source = (root / candidate.video_path).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"candidate footage for '{shot_id}' escapes the project folder") from exc
        if not source.is_file():
            raise ValueError(f"candidate source video does not exist for '{shot_id}': {source}")
        if candidate.source_end_seconds <= candidate.source_start_seconds:
            raise ValueError(f"candidate interval is invalid for '{shot_id}'")
        shot = shots[shot_id]
        shot.footage_path = candidate.video_path
        shot.footage_start_seconds = candidate.source_start_seconds
        shot.footage_end_seconds = candidate.source_end_seconds
        shot.assignment_status = "approved"

    for shot_id in rejected:
        shot = shots[shot_id]
        shot.footage_path = None
        shot.footage_start_seconds = None
        shot.footage_end_seconds = None
        shot.assignment_status = "unassigned"

    all_assigned = all(shot.assignment_status != "unassigned" for shot in shots.values())
    plan.planning_stage = "footage_mapped" if all_assigned else "storyboard_ready"
    plan.approved = False
    plan_file.write_text(json.dumps(plan.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    register_artifact(root, plan_file)
    return plan, plan_file
