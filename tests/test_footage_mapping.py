import hashlib
import json

import pytest

from video_edit_agent.artifact_tracking import register_artifact
from video_edit_agent.editing.footage_mapping import suggest_storyboard_footage
from video_edit_agent.editing.footage_schemas import ProjectFootageSuggestions
from video_edit_agent.editing.intent_schemas import EditingPlan
from video_edit_agent.editing.planning_schemas import (
    CreativeBrief,
    PlannedScene,
    StoryboardShot,
    StoryTreatment,
    VideoProjectPlan,
)
from video_edit_agent.understanding.schemas import (
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
    from video_edit_agent.editing.footage_mapping import _scene_text

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
    assert candidate.action_progression == ["Hiker approaches the viewpoint"]
    assert candidate.dynamics_confidence == 0.9
    assert candidate.subject_motion == "approaching"
    assert candidate.camera_motion == "static"
    assert candidate.primary_subject == "mountain lake"


def _write_footage_artifacts(tmp_path, plan, understanding):
    (tmp_path / "clip.mp4").write_bytes(b"test video placeholder")
    artifact_dir = tmp_path / ".video-agent"
    artifact_dir.mkdir(exist_ok=True)
    plan_path = artifact_dir / "video-project-plan.json"
    understanding_path = artifact_dir / "understanding.json"
    plan_path.write_text(json.dumps(plan.model_dump(mode="json")), encoding="utf-8")
    understanding_path.write_text(
        json.dumps(understanding.model_dump(mode="json")), encoding="utf-8"
    )
    register_artifact(tmp_path, plan_path)
    register_artifact(tmp_path, understanding_path)


def test_suggest_storyboard_footage_boosts_chapter_picks_but_keeps_outside_matches(
    tmp_path,
):
    plan = _project_plan(tmp_path)
    source_scenes = _understanding(tmp_path).scenes
    chapter_pick = source_scenes[1].scene_id
    plan = plan.model_copy(
        update={
            "scenes": [plan.scenes[0].model_copy(update={"candidate_scene_ids": [chapter_pick]})]
        }
    )
    understanding = _understanding(tmp_path)
    _write_footage_artifacts(tmp_path, plan, understanding)

    report = suggest_storyboard_footage(tmp_path, max_candidates_per_shot=2)

    candidates = report.suggestions[0].candidates
    chapter_candidate = next(item for item in candidates if item.source_scene_id == chapter_pick)
    outside_candidate = next(item for item in candidates if item.source_scene_id != chapter_pick)
    assert "Chapter pick" in " ".join(chapter_candidate.reasons)
    assert chapter_candidate.score > chapter_candidate.lexical_score
    assert outside_candidate.source_scene_id == source_scenes[0].scene_id


def test_suggest_storyboard_footage_always_includes_must_include_scenes(
    tmp_path,
):
    plan = _project_plan(tmp_path)
    source_scenes = _understanding(tmp_path).scenes
    must_include = source_scenes[1].scene_id
    plan = plan.model_copy(
        update={
            "scenes": [
                plan.scenes[0].model_copy(
                    update={
                        "candidate_scene_ids": [source_scenes[0].scene_id],
                        "must_include_scene_ids": [must_include],
                    }
                )
            ]
        }
    )
    _write_footage_artifacts(tmp_path, plan, _understanding(tmp_path))

    report = suggest_storyboard_footage(tmp_path, max_candidates_per_shot=1)

    candidates = report.suggestions[0].candidates
    assert len(candidates) == 2
    required = next(item for item in candidates if item.source_scene_id == must_include)
    assert "must-include" in " ".join(required.reasons).lower()


def test_suggest_storyboard_footage_discourages_reusing_previous_top_pick(tmp_path):
    plan = _project_plan(tmp_path)
    first_scene = plan.scenes[0]
    first_shot = first_scene.shots[0].model_copy(update={"target_duration_seconds": 4})
    first_scene = first_scene.model_copy(
        update={"target_duration_seconds": 4, "shots": [first_shot]}
    )
    second_shot = first_shot.model_copy(
        update={"shot_id": "shot-002", "scene_id": "scene-02", "sequence_index": 1}
    )
    second_scene = first_scene.model_copy(
        update={
            "scene_id": "scene-02",
            "sequence_index": 2,
            "title": "A second reveal",
            "target_duration_seconds": 4,
            "shots": [second_shot],
        }
    )
    plan = plan.model_copy(update={"scenes": [first_scene, second_scene]})

    understanding = _understanding(tmp_path)
    first_source = understanding.scenes[0]
    alternative = first_source.model_copy(
        update={
            "scene_id": first_source.scene_id.rsplit("-", 1)[0] + "-0002",
            "start_seconds": 12,
            "end_seconds": 20,
        }
    )
    understanding = understanding.model_copy(update={"scenes": [first_source, alternative]})
    _write_footage_artifacts(tmp_path, plan, understanding)

    report = suggest_storyboard_footage(tmp_path, max_candidates_per_shot=2)

    first_pick = report.suggestions[0].candidates[0].source_scene_id
    second_candidates = report.suggestions[1].candidates
    assert second_candidates[0].source_scene_id != first_pick
    reused = next(item for item in second_candidates if item.source_scene_id == first_pick)
    assert "Reuse discouraged" in " ".join(reused.reasons)


def test_suggest_storyboard_footage_fails_if_must_include_scene_is_unavailable(tmp_path):
    plan = _project_plan(tmp_path)
    plan = plan.model_copy(
        update={
            "scenes": [
                plan.scenes[0].model_copy(
                    update={"must_include_scene_ids": ["missing-source-scene"]}
                )
            ]
        }
    )
    _write_footage_artifacts(tmp_path, plan, _understanding(tmp_path))

    with pytest.raises(ValueError, match="must-include scenes are missing"):
        suggest_storyboard_footage(tmp_path)



def test_footage_feedback_adjusts_similar_shot_candidate_scores(tmp_path):
    from video_edit_agent.artifact_tracking import register_artifact
    from video_edit_agent.editing.footage_feedback import (
        FootageFeedbackEntry,
        ProjectFootageFeedback,
        shot_feedback_tokens,
    )

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
    baseline = suggest_storyboard_footage(tmp_path)
    preferred = baseline.suggestions[0].candidates[0]
    shot = _project_plan(tmp_path).scenes[0].shots[0]
    feedback_path = artifact_dir / "footage-feedback.json"
    feedback_path.write_text(
        ProjectFootageFeedback(
            project_path=str(tmp_path.resolve()),
            entries=[
                FootageFeedbackEntry(
                    shot_tokens=shot_feedback_tokens(shot),
                    shot_scale=shot.shot_scale,
                    visual_role=shot.visual_role,
                    rejected_scene_ids=[preferred.source_scene_id],
                )
            ],
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    register_artifact(tmp_path, feedback_path)

    learned = suggest_storyboard_footage(tmp_path)
    candidates = {
        item.source_scene_id: item for item in learned.suggestions[0].candidates
    }

    assert candidates[preferred.source_scene_id].score == preferred.score - 35
    assert any(
        "previously rejected" in reason
        for reason in candidates[preferred.source_scene_id].reasons
    )
    assert learned.source_feedback == ".video-agent/footage-feedback.json"
