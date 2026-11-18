import hashlib
import json

import pytest

from travel_video_agent.artifact_tracking import register_artifact
from travel_video_agent.editing.footage_mapping import suggest_storyboard_footage
from travel_video_agent.editing.footage_schemas import ProjectFootageSuggestions
from travel_video_agent.editing.intent_schemas import EditingPlan
from travel_video_agent.editing.planning_schemas import (
    CreativeBrief,
    PlannedScene,
    StoryboardShot,
    StoryTreatment,
    VideoProjectPlan,
)
from travel_video_agent.understanding.schemas import (
    FrameObservation,
    SceneDynamics,
    VideoScene,
    VideoUnderstanding,
)


def _project_plan(root):
    editing = EditingPlan(
        source_prompt="Create a cinematic mountain travel film",
        target_duration_seconds=8,
        style="cinematic",
        priorities=["scenic_landscapes"],
    )
    brief = CreativeBrief(
        source_prompt=editing.source_prompt,
        target_duration_seconds=8,
        style="cinematic",
        aspect_ratio="16:9",
        resolution="1080p",
        pacing="moderate",
        creative_priorities=["scenic_landscapes"],
    )
    shot = StoryboardShot(
        shot_id="shot-001",
        scene_id="scene-01",
        sequence_index=1,
        purpose="Reveal a dramatic mountain landscape",
        visual_description="A wide mountain panorama beside a turquoise lake",
        shot_scale="wide",
        visual_role="establishing",
        target_duration_seconds=8,
    )
    scene = PlannedScene(
        scene_id="scene-01",
        sequence_index=1,
        title="The first reveal",
        purpose="Establish the alpine setting",
        narrative_function="hook",
        target_duration_seconds=8,
        shots=[shot],
    )
    return VideoProjectPlan(
        project_path=str(root),
        brief=brief,
        editing_preferences=editing,
        story_treatment=StoryTreatment(
            working_title="Alpine Light",
            logline="A journey into a mountain landscape.",
            narrative_arc=["Arrival", "Discovery"],
            emotional_journey="Wonder and calm",
        ),
        scenes=[scene],
        planning_stage="storyboard_ready",
    )


def _understanding(root):
    relative_path = "clip.mp4"
    video_key = hashlib.sha256(relative_path.encode()).hexdigest()[:12]
    return VideoUnderstanding(
        project_path=str(root),
        video_count=1,
        scene_count=2,
        model="test-model",
        scenes=[
            VideoScene(
                scene_id=f"{video_key}-scene-0001",
                start_seconds=0,
                end_seconds=12,
                summary="Wide mountain panorama beside a turquoise alpine lake.",
                story_function="establishing",
                dynamics=SceneDynamics(
                    temporal_summary="A hiker walks toward the mountain viewpoint.",
                    action_progression=["Hiker approaches the viewpoint"],
                    subject_motion="approaching",
                    camera_motion="static",
                    confidence=0.9,
                ),
                observations=[
                    FrameObservation(
                        timestamp_seconds=2,
                        frame_path=".video-agent/frames/frame-1.jpg",
                        description="A wide mountain panorama and turquoise lake.",
                        objects=["mountain", "lake"],
                        shot_scale="wide",
                        visual_role="establishing",
                        setting="alpine landscape",
                        primary_subject="mountain lake",
                    )
                ],
            ),
            VideoScene(
                scene_id=f"{video_key}-scene-0002",
                start_seconds=12,
                end_seconds=20,
                summary="Close-up of food being prepared at a busy market.",
                story_function="detail",
                observations=[
                    FrameObservation(
                        timestamp_seconds=15,
                        frame_path=".video-agent/frames/frame-2.jpg",
                        description="A close-up of food preparation.",
                        objects=["food", "market"],
                        shot_scale="close_up",
                        visual_role="detail",
                    )
                ],
            ),
        ],
    )


def test_suggest_storyboard_footage_ranks_matching_scene_and_does_not_assign(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"test video placeholder")
    artifact_dir = tmp_path / ".video-agent"
    artifact_dir.mkdir()
    plan_path = artifact_dir / "video-project-plan.json"
    understanding_path = artifact_dir / "understanding.json"
    plan_path.write_text(
        json.dumps(_project_plan(tmp_path).model_dump(mode="json")), encoding="utf-8"
    )
    understanding_path.write_text(
        json.dumps(_understanding(tmp_path).model_dump(mode="json")), encoding="utf-8"
    )
    register_artifact(tmp_path, plan_path)
    register_artifact(tmp_path, understanding_path)

    report = suggest_storyboard_footage(tmp_path, max_candidates_per_shot=2)

    assert isinstance(report, ProjectFootageSuggestions)
    assert report.matched_shot_count == 1
    assert report.candidate_count == 2
    candidates = report.suggestions[0].candidates
    assert candidates[0].source_scene_id.endswith("scene-0001")
    assert candidates[0].score > candidates[1].score
    assert candidates[0].source_start_seconds == 0
    assert candidates[0].source_end_seconds == 12
    assert report.review_required is True
    assert report.assignments_performed is False


def test_suggest_storyboard_footage_rejects_invalid_candidate_limit(tmp_path):
    with pytest.raises(ValueError, match="between 1 and 10"):
        suggest_storyboard_footage(tmp_path, max_candidates_per_shot=0)


def test_suggest_storyboard_footage_uses_hybrid_semantic_ranking(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"test video placeholder")
    artifact_dir = tmp_path / ".video-agent"
    artifact_dir.mkdir()
    plan_path = artifact_dir / "video-project-plan.json"
    understanding_path = artifact_dir / "understanding.json"
    plan_path.write_text(
        json.dumps(_project_plan(tmp_path).model_dump(mode="json")), encoding="utf-8"
    )
    understanding_path.write_text(
        json.dumps(_understanding(tmp_path).model_dump(mode="json")), encoding="utf-8"
    )
    register_artifact(tmp_path, plan_path)
    register_artifact(tmp_path, understanding_path)

    def fake_embedder(texts, model):
        assert model == "test-embedding-model"
        assert len(texts) == 3
        return [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    report = suggest_storyboard_footage(
        tmp_path,
        max_candidates_per_shot=2,
        embedding_model="test-embedding-model",
        embedder=fake_embedder,
    )

    candidates = report.suggestions[0].candidates
    assert report.matching_method == "hybrid"
    assert report.embedding_model == "test-embedding-model"
    assert candidates[0].match_method == "hybrid"
    assert candidates[0].semantic_similarity == pytest.approx(1.0)
    assert candidates[1].semantic_similarity == pytest.approx(0.5)
    assert candidates[0].lexical_score > 0
    assert candidates[0].score > candidates[1].score
    assert report.assignments_performed is False


def test_suggest_storyboard_footage_rejects_invalid_embedding_batch(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"test video placeholder")
    artifact_dir = tmp_path / ".video-agent"
    artifact_dir.mkdir()
    plan_path = artifact_dir / "video-project-plan.json"
    understanding_path = artifact_dir / "understanding.json"
    plan_path.write_text(
        json.dumps(_project_plan(tmp_path).model_dump(mode="json")), encoding="utf-8"
    )
    understanding_path.write_text(
        json.dumps(_understanding(tmp_path).model_dump(mode="json")), encoding="utf-8"
    )
    register_artifact(tmp_path, plan_path)
    register_artifact(tmp_path, understanding_path)

    with pytest.raises(ValueError, match="different number of vectors"):
        suggest_storyboard_footage(
            tmp_path,
            embedding_model="test-embedding-model",
            embedder=lambda texts, model: [],
        )


def test_scene_text_includes_temporal_action_and_camera_evidence(tmp_path):
    from travel_video_agent.editing.footage_mapping import _scene_text

    scene = _understanding(tmp_path).scenes[0]
    text = _scene_text(scene).lower()
    assert "hiker approaches the viewpoint" in text
    assert "approaching" in text
    assert "static" in text


def test_footage_candidates_include_temporal_and_composition_evidence(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"test video placeholder")
    artifact_dir = tmp_path / ".video-agent"
    artifact_dir.mkdir()
    plan_path = artifact_dir / "video-project-plan.json"
    understanding_path = artifact_dir / "understanding.json"
    plan_path.write_text(
        json.dumps(_project_plan(tmp_path).model_dump(mode="json")), encoding="utf-8"
    )
    understanding_path.write_text(
        json.dumps(_understanding(tmp_path).model_dump(mode="json")), encoding="utf-8"
    )
    register_artifact(tmp_path, plan_path)
    register_artifact(tmp_path, understanding_path)

    report = suggest_storyboard_footage(tmp_path)
    candidate = report.suggestions[0].candidates[0]

    assert candidate.temporal_summary == "A hiker walks toward the mountain viewpoint."
    assert candidate.action_progression == ["The hiker approaches the viewpoint"]
    assert candidate.dynamics_confidence == 0.9
    assert candidate.subject_motion == "approaching"
    assert candidate.camera_motion == "static"
    assert candidate.primary_subject == "mountain lake"
