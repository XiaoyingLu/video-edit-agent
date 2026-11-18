"""Deterministic, explainable matching between storyboard shots and analyzed scenes."""

import hashlib
import math
import re
from collections.abc import Callable
from pathlib import Path

from ..artifact_tracking import validate_artifact
from ..media import discover_videos
from ..understanding.schemas import VideoUnderstanding
from .embeddings import ollama_embed_texts
from .footage_schemas import (
    FootageMatchCandidate,
    ProjectFootageSuggestions,
    ShotFootageSuggestions,
)
from .planning_schemas import VideoProjectPlan

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "with",
    "show",
    "shot",
    "view",
    "scene",
    "camera",
    "footage",
}


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 2 and token not in _STOP_WORDS
    }


def _scene_text(scene) -> str:
    parts = [scene.summary]
    for observation in scene.observations:
        parts.extend(
            [
                observation.description,
                observation.setting,
                observation.primary_subject,
                *observation.objects,
                *observation.activities,
            ]
        )
    return " ".join(parts)


def _candidate_score(
    shot, scene, semantic_similarity: float | None = None
) -> tuple[float, list[str], float]:
    shot_terms = _tokens(f"{shot.purpose} {shot.visual_description}")
    scene_terms = _tokens(_scene_text(scene))
    overlap = shot_terms & scene_terms
    coverage = len(overlap) / len(shot_terms) if shot_terms else 0.0
    scale_match = shot.shot_scale != "unknown" and any(
        item.shot_scale == shot.shot_scale for item in scene.observations
    )
    role_match = shot.visual_role != "unknown" and (
        scene.story_function == shot.visual_role
        or any(item.visual_role == shot.visual_role for item in scene.observations)
    )
    lexical_score = min(
        100.0,
        coverage * 70.0 + (15.0 if scale_match else 0.0) + (15.0 if role_match else 0.0),
    )
    if semantic_similarity is None:
        score = lexical_score
    else:
        score = (
            max(0.0, min(1.0, semantic_similarity)) * 45.0
            + coverage * 35.0
            + (10.0 if scale_match else 0.0)
            + (10.0 if role_match else 0.0)
        )
    reasons = []
    if semantic_similarity is not None:
        reasons.append(f"Semantic similarity: {semantic_similarity:.2f}.")
    if overlap:
        reasons.append(f"Description matches: {', '.join(sorted(overlap))}.")
    if scale_match:
        reasons.append(f"Shot scale matches: {shot.shot_scale.replace('_', ' ')}.")
    if role_match:
        reasons.append(f"Editorial role matches: {shot.visual_role.replace('_', ' ')}.")
    if not reasons:
        reasons.append("No strong semantic or editorial match; shown as a low-confidence fallback.")
    return round(score, 1), reasons, round(lexical_score, 1)


def suggest_storyboard_footage(
    folder: Path,
    *,
    plan_path: Path | None = None,
    understanding_path: Path | None = None,
    max_candidates_per_shot: int = 3,
    embedding_model: str | None = None,
    embedder: Callable[[list[str], str], list[list[float]]] | None = None,
) -> ProjectFootageSuggestions:
    """Rank analyzed source scenes for each storyboard shot without assigning footage."""
    if not 1 <= max_candidates_per_shot <= 10:
        raise ValueError("max_candidates_per_shot must be between 1 and 10")

    root = folder.expanduser().resolve()
    artifact_dir = root / ".video-agent"
    plan_file = (plan_path or artifact_dir / "video-project-plan.json").expanduser().resolve()
    understanding_file = (
        (understanding_path or artifact_dir / "understanding.json").expanduser().resolve()
    )
    for path, label in (
        (plan_file, "video project plan"),
        (understanding_file, "video understanding"),
    ):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{label} must be inside the project folder") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; generate the {label} first.")
        validate_artifact(root, path)

    plan = VideoProjectPlan.model_validate_json(plan_file.read_text(encoding="utf-8"))
    if Path(plan.project_path).resolve() != root:
        raise ValueError("video project plan belongs to a different project folder")
    if plan.planning_stage != "storyboard_ready":
        raise ValueError("footage suggestions require a storyboard_ready plan")

    understanding = VideoUnderstanding.model_validate_json(
        understanding_file.read_text(encoding="utf-8")
    )
    video_paths = {}
    for path in discover_videos(root):
        relative_path = path.relative_to(root).as_posix()
        video_key = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]
        video_paths[video_key] = relative_path

    scenes = []
    for source_scene in understanding.scenes:
        video_key = source_scene.scene_id.split("-scene-", maxsplit=1)[0]
        video_path = video_paths.get(video_key)
        if video_path is not None and source_scene.end_seconds > source_scene.start_seconds:
            scenes.append((video_path, source_scene))

    shot_items = [shot for planned_scene in plan.scenes for shot in planned_scene.shots]
    semantic_scores: dict[tuple[str, str], float] = {}
    if embedding_model:
        texts = [
            f"Storyboard intent: {shot.purpose}. Visual requirements: {shot.visual_description}"
            for shot in shot_items
        ] + [_scene_text(source_scene) for _, source_scene in scenes]
        vectors = (embedder or ollama_embed_texts)(texts, embedding_model)
        if len(vectors) != len(texts):
            raise ValueError(
                "embedding provider returned a different number of vectors than requested"
            )
        shot_vectors = vectors[: len(shot_items)]
        scene_vectors = vectors[len(shot_items) :]
        if any(len(vector) != len(shot_vectors[0]) for vector in vectors):
            raise ValueError("embedding provider returned inconsistent vector dimensions")
        for shot_index, shot_vector in enumerate(shot_vectors):
            shot_norm = math.sqrt(sum(value * value for value in shot_vector))
            for scene_index, scene_vector in enumerate(scene_vectors):
                scene_norm = math.sqrt(sum(value * value for value in scene_vector))
                similarity = (
                    sum(a * b for a, b in zip(shot_vector, scene_vector, strict=True))
                    / (shot_norm * scene_norm)
                    if shot_norm and scene_norm
                    else 0.0
                )
                semantic_scores[
                    (shot_items[shot_index].shot_id, scenes[scene_index][1].scene_id)
                ] = max(0.0, min(1.0, (similarity + 1.0) / 2.0))

    suggestions = []
    candidate_count = 0
    matched_shot_count = 0
    for planned_scene in plan.scenes:
        for shot in planned_scene.shots:
            ranked = []
            for video_path, source_scene in scenes:
                semantic_similarity = semantic_scores.get((shot.shot_id, source_scene.scene_id))
                score, reasons, lexical_score = _candidate_score(
                    shot, source_scene, semantic_similarity
                )
                ranked.append(
                    FootageMatchCandidate(
                        video_path=video_path,
                        source_scene_id=source_scene.scene_id,
                        source_start_seconds=round(source_scene.start_seconds, 3),
                        source_end_seconds=round(source_scene.end_seconds, 3),
                        score=score,
                        lexical_score=lexical_score,
                        semantic_similarity=semantic_similarity,
                        match_method="hybrid" if semantic_similarity is not None else "lexical",
                        scene_summary=source_scene.summary,
                        reasons=reasons,
                    )
                )
            ranked.sort(
                key=lambda item: (
                    -item.score,
                    item.video_path,
                    item.source_start_seconds,
                    item.source_scene_id,
                )
            )
            candidates = ranked[:max_candidates_per_shot]
            candidate_count += len(candidates)
            matched_shot_count += bool(candidates)
            suggestions.append(
                ShotFootageSuggestions(
                    shot_id=shot.shot_id,
                    scene_id=shot.scene_id,
                    purpose=shot.purpose,
                    visual_description=shot.visual_description,
                    target_duration_seconds=shot.target_duration_seconds,
                    candidates=candidates,
                )
            )

    return ProjectFootageSuggestions(
        project_path=str(root),
        source_plan=plan_file.relative_to(root).as_posix(),
        source_understanding=understanding_file.relative_to(root).as_posix(),
        suggestions=suggestions,
        candidate_count=candidate_count,
        matched_shot_count=matched_shot_count,
        embedding_model=embedding_model,
        matching_method="hybrid" if embedding_model else "lexical",
    )
