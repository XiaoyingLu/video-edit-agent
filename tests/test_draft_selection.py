import json

import pytest

from travel_video_agent.editing.draft_pipeline import build_draft_selection
from travel_video_agent.editing.intent_schemas import EditingPlan


def _write_artifacts(root, candidates, scenes=None, transcript=None):
    artifact = root / ".video-agent"
    artifact.mkdir(parents=True, exist_ok=True)
    plan = EditingPlan(
        source_prompt="Make a scenic travel video",
        target_duration_seconds=8,
        priorities=["scenic_landscapes", "memorable_moments"],
    )
    (artifact / "editing-plan.json").write_text(plan.model_dump_json(), encoding="utf-8")
    (artifact / "highlights.json").write_text(
        json.dumps(
            {
                "project_path": str(root),
                "video_count": 1,
                "candidate_count": len(candidates),
                "candidates": candidates,
            }
        ),
        encoding="utf-8",
    )
    if scenes is not None:
        (artifact / "understanding.json").write_text(
            json.dumps(
                {
                    "project_path": str(root),
                    "video_count": 1,
                    "scene_count": len(scenes),
                    "model": "test",
                    "scenes": scenes,
                }
            ),
            encoding="utf-8",
        )
    if transcript is not None:
        (artifact / "transcription.json").write_text(json.dumps(transcript), encoding="utf-8")


def test_draft_selection_ranks_by_visual_score_and_scene_relevance(tmp_path):
    video_path = "clip-a.mp4"
    candidates = [
        {
            "candidate_id": "low",
            "video_path": video_path,
            "start_seconds": 0,
            "end_seconds": 5,
            "duration_seconds": 5,
            "score": 60,
            "signals": {"motion": 0.1, "sharpness": 0.7, "exposure": 0.7, "saturation": 0.5},
            "reasons": ["Low motion: calm scenic shot."],
            "scene_summary": "A quiet ocean landscape at sunset",
            "review_required": True,
        },
        {
            "candidate_id": "high",
            "video_path": video_path,
            "start_seconds": 5,
            "end_seconds": 10,
            "duration_seconds": 5,
            "score": 90,
            "signals": {"motion": 0.5, "sharpness": 0.9, "exposure": 0.9, "saturation": 0.8},
            "reasons": [],
            "scene_summary": "People walking through a city street",
            "review_required": True,
        },
    ]
    _write_artifacts(tmp_path, candidates)
    plan = EditingPlan(
        source_prompt="Make a scenic travel video",
        target_duration_seconds=8,
        priorities=["scenic_landscapes", "memorable_moments"],
    )
    draft = build_draft_selection(tmp_path, plan)
    assert draft.selected_clip_count == 2
    assert draft.selected_duration_seconds == 8
    assert draft.clips[0].clip_id == "high"
    assert draft.clips[0].selected is True
    assert draft.clips[1].selected is True
    assert draft.approval_required is True
    assert draft.rendering_performed is False
    assert draft.clips[1].source_end_seconds == 3


def test_draft_selection_requires_highlight_artifact(tmp_path):
    with pytest.raises(FileNotFoundError, match="highlights.json"):
        build_draft_selection(tmp_path, EditingPlan(source_prompt="Make a video"))


def test_draft_selection_never_exceeds_target_duration(tmp_path):
    candidates = [
        {
            "candidate_id": "long",
            "video_path": "clip.mp4",
            "start_seconds": 0,
            "end_seconds": 30,
            "duration_seconds": 30,
            "score": 95,
            "signals": {"motion": 0.5, "sharpness": 0.9, "exposure": 0.9, "saturation": 0.8},
            "reasons": [],
            "scene_summary": "Mountain landscape",
            "review_required": True,
        }
    ]
    _write_artifacts(tmp_path, candidates)
    plan = EditingPlan(source_prompt="Make a video", target_duration_seconds=10)
    draft = build_draft_selection(tmp_path, plan)
    assert draft.selected_duration_seconds == 10
    assert draft.clips[0].source_end_seconds == 10


def test_draft_selection_carries_cinematic_scene_metadata(tmp_path):
    import hashlib

    video_path = "clip-a.mp4"
    video_key = hashlib.sha256(video_path.encode("utf-8")).hexdigest()[:12]
    candidates = [
        {
            "candidate_id": "scene-1",
            "video_path": video_path,
            "start_seconds": 0,
            "end_seconds": 5,
            "duration_seconds": 5,
            "score": 85,
            "signals": {"motion": 0.2, "sharpness": 0.8, "exposure": 0.8, "saturation": 0.5},
            "reasons": [],
            "scene_summary": "A wide mountain lake at sunrise",
            "review_required": True,
        }
    ]
    scenes = [
        {
            "scene_id": f"{video_key}-scene-0001",
            "start_seconds": 0,
            "end_seconds": 5,
            "summary": "A wide mountain lake at sunrise",
            "story_function": "establishing",
            "dominant_mood": "serene",
            "observations": [
                {
                    "timestamp_seconds": 1,
                    "frame_path": "frame-1.jpg",
                    "description": "A wide mountain lake.",
                    "objects": ["lake", "mountain"],
                    "activities": [],
                    "confidence": 0.9,
                    "shot_scale": "wide",
                    "visual_role": "establishing",
                    "mood": "serene",
                    "setting": "alpine lake",
                    "primary_subject": "mountain lake",
                },
                {
                    "timestamp_seconds": 3,
                    "frame_path": "frame-2.jpg",
                    "description": "Sunrise over the lake.",
                    "objects": ["lake", "sun"],
                    "activities": [],
                    "confidence": 0.9,
                    "shot_scale": "unknown",
                    "visual_role": "establishing",
                    "mood": "serene",
                    "setting": "alpine lake",
                    "primary_subject": "mountain lake",
                },
            ],
        }
    ]
    _write_artifacts(tmp_path, candidates, scenes=scenes)
    plan = EditingPlan(
        source_prompt="Create a scenic travel video",
        target_duration_seconds=5,
        priorities=["scenic_landscapes"],
    )

    draft = build_draft_selection(tmp_path, plan)

    clip = draft.clips[0]
    assert clip.shot_scale == "wide"
    assert clip.visual_role == "establishing"
    assert clip.mood == "serene"
    assert clip.setting == "alpine lake"
    assert clip.primary_subject == "mountain lake"
