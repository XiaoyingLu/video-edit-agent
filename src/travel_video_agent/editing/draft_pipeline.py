"""Deterministic context-aware shot selection for a natural-language edit plan."""

import hashlib
from pathlib import Path

from ..artifact_tracking import validate_artifact
from ..highlights.schemas import ProjectHighlights
from ..transcription.schemas import ProjectTranscription
from ..understanding.schemas import VideoUnderstanding
from .draft_schemas import DraftClip, ProjectDraftSelection
from .intent_schemas import EditingPlan

_PRIORITY_TERMS = {
    "scenic_landscapes": {
        "landscape",
        "mountain",
        "ocean",
        "beach",
        "lake",
        "forest",
        "waterfall",
        "sunset",
        "sky",
        "coast",
        "scenic",
    },
    "city_highlights": {"city", "street", "traffic", "skyline", "downtown", "urban", "square"},
    "people_and_activities": {
        "people",
        "person",
        "walking",
        "hiking",
        "cycling",
        "activity",
        "crowd",
        "family",
    },
    "food_and_culture": {"food", "restaurant", "market", "festival", "culture", "cooking", "dish"},
    "architecture": {"building", "architecture", "church", "temple", "bridge", "museum", "castle"},
    "memorable_moments": {
        "smile",
        "celebration",
        "sunset",
        "view",
        "landmark",
        "moment",
        "beautiful",
        "panorama",
    },
    "chronological_story": set(),
}


def _scene_for_candidate(candidate, scenes):
    overlaps = [
        scene
        for scene in scenes
        if scene.start_seconds < candidate.end_seconds
        and scene.end_seconds > candidate.start_seconds
    ]
    if not overlaps:
        return None
    return max(
        overlaps,
        key=lambda scene: (
            min(candidate.end_seconds, scene.end_seconds)
            - max(candidate.start_seconds, scene.start_seconds)
        ),
    )


def _transcript_excerpt(video, start: float, end: float) -> str | None:
    if video is None:
        return None
    parts = [
        segment.text
        for segment in video.segments
        if segment.start_seconds < end and segment.end_seconds > start and segment.text
    ]
    return " ".join(parts)[:500] or None


def _relevance(summary: str | None, priorities: list[str]) -> tuple[float, list[str]]:
    text = (summary or "").lower()
    if not text:
        return 50.0, ["No scene description available; using neutral relevance."]
    matched: list[str] = []
    for priority in priorities:
        terms = _PRIORITY_TERMS.get(priority, set())
        if terms and any(term in text for term in terms):
            matched.append(priority)
    if not priorities or priorities == ["memorable_moments"]:
        score = 70.0 if matched else 50.0
    else:
        score = min(100.0, 45.0 + 55.0 * len(matched) / len(priorities))
    reasons = (
        [f"Scene description matches: {', '.join(matched)}."]
        if matched
        else ["No explicit priority keyword match in the scene description."]
    )
    return score, reasons


def build_draft_selection(
    folder: Path,
    editing_plan: EditingPlan | None = None,
    *,
    plan_path: Path | None = None,
) -> ProjectDraftSelection:
    """Rank and select scene intervals toward the requested duration.

    Existing analysis artifacts are read only. The resulting selection is a review
    artifact, not a renderer timeline. Quiet scenic scenes are never excluded solely
    because they contain no speech or have low motion.
    """
    root = folder.expanduser().resolve()
    artifact_dir = root / ".video-agent"
    plan_file = (plan_path or artifact_dir / "editing-plan.json").expanduser().resolve()
    if editing_plan is None:
        if not plan_file.is_file():
            raise FileNotFoundError(f"Missing {plan_file}; run 'video-agent plan' first.")
        validate_artifact(root, plan_file)
        editing_plan = EditingPlan.model_validate_json(plan_file.read_text(encoding="utf-8"))

    highlights_file = artifact_dir / "highlights.json"
    if not highlights_file.is_file():
        raise FileNotFoundError(
            f"Missing {highlights_file}; run 'video-agent find-highlights' first."
        )
    validate_artifact(root, highlights_file)
    highlights = ProjectHighlights.model_validate_json(highlights_file.read_text(encoding="utf-8"))
    understanding_file = artifact_dir / "understanding.json"
    if understanding_file.is_file():
        validate_artifact(root, understanding_file)
        understanding = VideoUnderstanding.model_validate_json(
            understanding_file.read_text(encoding="utf-8")
        )
    else:
        understanding = None
    transcription_file = artifact_dir / "transcription.json"
    if transcription_file.is_file():
        validate_artifact(root, transcription_file)
        transcription = ProjectTranscription.model_validate_json(
            transcription_file.read_text(encoding="utf-8")
        )
    else:
        transcription = None

    scenes_by_video: dict[str, list] = {}
    if understanding is not None:
        for scene in understanding.scenes:
            video_key = scene.scene_id.split("-scene-", maxsplit=1)[0]
            scenes_by_video.setdefault(video_key, []).append(scene)
    transcripts = {video.path: video for video in transcription.videos} if transcription else {}

    candidates: list[DraftClip] = []
    for candidate in highlights.candidates:
        video_key = hashlib.sha256(candidate.video_path.encode("utf-8")).hexdigest()[:12]
        scene = _scene_for_candidate(candidate, scenes_by_video.get(video_key, []))
        summary = candidate.scene_summary or (scene.summary if scene else None)
        relevance, reasons = _relevance(summary, editing_plan.priorities)
        excerpt = _transcript_excerpt(
            transcripts.get(candidate.video_path),
            candidate.start_seconds,
            candidate.end_seconds,
        )
        combined = round(candidate.score * 0.65 + relevance * 0.35, 1)
        if (
            excerpt
            and editing_plan.priorities
            and (
                "people_and_activities" in editing_plan.priorities
                or "memorable_moments" in editing_plan.priorities
            )
        ):
            combined = min(100.0, combined + 5.0)
            reasons.append("Overlapping speech may add human context to this moment.")
        reasons.extend(candidate.reasons[:2])
        candidates.append(
            DraftClip(
                clip_id=candidate.candidate_id,
                video_path=candidate.video_path,
                source_start_seconds=candidate.start_seconds,
                source_end_seconds=candidate.end_seconds,
                duration_seconds=candidate.duration_seconds,
                highlight_score=candidate.score,
                relevance_score=relevance,
                combined_score=combined,
                scene_summary=summary,
                transcript_excerpt=excerpt,
                shot_scale=scene.observations[0].shot_scale if scene and scene.observations else "unknown",
                visual_role=scene.story_function if scene else "unknown",
                mood=scene.dominant_mood if scene else "unknown",
                setting=(
                    next(
                        (item.setting for item in scene.observations if item.setting),
                        "",
                    )
                    if scene
                    else ""
                ),
                primary_subject=(
                    next(
                        (item.primary_subject for item in scene.observations if item.primary_subject),
                        "",
                    )
                    if scene
                    else ""
                ),
                reasons=reasons,
            )
        )

    if "chronological_story" in editing_plan.priorities:
        candidates.sort(key=lambda item: (item.video_path, item.source_start_seconds))
    else:
        candidates.sort(
            key=lambda item: (-item.combined_score, item.video_path, item.source_start_seconds)
        )

    remaining = editing_plan.target_duration_seconds
    selected_ids: set[str] = set()
    selected_duration = 0.0
    selected_clips: list[DraftClip] = []
    for clip in candidates:
        if remaining < 1.0:
            break
        if clip.duration_seconds < 1.0:
            continue
        duration = min(clip.duration_seconds, remaining)
        selected = clip.model_copy(
            update={
                "source_end_seconds": round(clip.source_start_seconds + duration, 3),
                "duration_seconds": round(duration, 3),
                "selected": True,
            }
        )
        selected_clips.append(selected)
        selected_ids.add(clip.clip_id)
        selected_duration += duration
        remaining -= duration

    output_clips = [
        next((item for item in selected_clips if item.clip_id == candidate.clip_id), candidate)
        for candidate in candidates
    ]
    return ProjectDraftSelection(
        project_path=str(root),
        source_artifacts=[
            str(plan_file.relative_to(root)) if plan_file.is_relative_to(root) else str(plan_file),
            ".video-agent/highlights.json",
            *([".video-agent/understanding.json"] if understanding is not None else []),
            *([".video-agent/transcription.json"] if transcription is not None else []),
        ],
        target_duration_seconds=editing_plan.target_duration_seconds,
        selected_duration_seconds=round(selected_duration, 3),
        selected_clip_count=len(selected_ids),
        candidates_considered=len(candidates),
        clips=output_clips,
    )
