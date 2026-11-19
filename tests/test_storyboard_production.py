from pathlib import Path

import pytest

from travel_video_agent.artifact_tracking import register_artifact
from travel_video_agent.editing.planning_schemas import VideoProjectPlan
from travel_video_agent.editing.storyboard_production import (
    approve_storyboard,
    build_storyboard_timeline,
    save_storyboard_timeline,
)
from travel_video_agent.editing.timeline_schemas import ProjectDraftTimeline


def _write_plan(root: Path, *, mapped: bool = True) -> Path:
    artifact_dir = root / ".video-agent"
    artifact_dir.mkdir(exist_ok=True)
    (root / "travel.mp4").write_bytes(b"local source placeholder")
    shots = [
        {
            "shot_id": "shot-001",
            "scene_id": "scene-01",
            "sequence_index": 1,
            "purpose": "Establish the mountain landscape",
            "visual_description": "A wide mountain lake at sunrise",
            "shot_scale": "wide",
            "visual_role": "establishing",
            "primary_subject": "mountain lake",
            "temporal_summary": "The camera tracks a hiker toward the lake.",
            "action_progression": ["The hiker approaches the lake viewpoint"],
            "dynamics_confidence": 0.9,
            "subject_motion": "approaching",
            "camera_motion": "tracking",
            "composition": "leading_lines",
            "target_duration_seconds": 4,
            "footage_path": "travel.mp4" if mapped else None,
            "footage_start_seconds": 0 if mapped else None,
            "footage_end_seconds": 8 if mapped else None,
            "assignment_status": "approved" if mapped else "unassigned",
        },
        {
            "shot_id": "shot-002",
            "scene_id": "scene-01",
            "sequence_index": 2,
            "purpose": "Reveal the shoreline detail",
            "visual_description": "Close-up of water moving over rocks",
            "shot_scale": "close_up",
            "visual_role": "detail",
            "target_duration_seconds": 3,
            "footage_path": "travel.mp4" if mapped else None,
            "footage_start_seconds": 10 if mapped else None,
            "footage_end_seconds": 12 if mapped else None,
            "assignment_status": "approved" if mapped else "unassigned",
        },
    ]
    plan = VideoProjectPlan(
        project_path=str(root.resolve()),
        brief={
            "source_prompt": "Create a 7-second cinematic travel film",
            "target_duration_seconds": 7,
            "style": "cinematic",
            "aspect_ratio": "16:9",
            "resolution": "1080p",
            "pacing": "moderate",
        },
        editing_preferences={
            "source_prompt": "Create a 7-second cinematic travel film",
            "target_duration_seconds": 7,
            "style": "cinematic",
            "pacing": "moderate",
            "transitions": {"default": "crossfade", "duration_seconds": 0.5},
        },
        story_treatment={
            "working_title": "Mountain Light",
            "logline": "A quiet mountain journey.",
            "narrative_arc": ["Arrival", "Discovery"],
            "emotional_journey": "Calm becomes awe.",
        },
        scenes=[
            {
                "scene_id": "scene-01",
                "sequence_index": 1,
                "title": "First Light",
                "purpose": "Introduce the mountain setting.",
                "narrative_function": "hook",
                "target_duration_seconds": 7,
                "shots": shots,
            }
        ],
        planning_stage="footage_mapped" if mapped else "storyboard_ready",
    )
    path = artifact_dir / "video-project-plan.json"
    path.write_text(plan.model_dump_json(), encoding="utf-8")
    register_artifact(root, path)
    (artifact_dir / "editing-plan.json").write_text(
        plan.editing_preferences.model_dump_json(), encoding="utf-8"
    )
    register_artifact(root, artifact_dir / "editing-plan.json")
    return path


def test_approve_storyboard_requires_all_footage_assignments(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path, mapped=False)

    with pytest.raises(ValueError, match="requires every shot to have an approved footage match"):
        approve_storyboard(tmp_path, plan_path=plan_path)


def test_approve_and_compile_storyboard_preserves_order_and_trims(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)

    plan, _ = approve_storyboard(tmp_path, plan_path=plan_path)
    timeline = build_storyboard_timeline(tmp_path, plan_path=plan_path)

    assert plan.approved is True
    assert plan.planning_stage == "production_ready"
    assert [clip.clip_id for clip in timeline.clips] == ["shot-001", "shot-002"]
    assert (timeline.clips[0].source_start_seconds, timeline.clips[0].source_end_seconds) == (
        2,
        6,
    )
    assert (timeline.clips[1].source_start_seconds, timeline.clips[1].source_end_seconds) == (
        10,
        12,
    )
    assert timeline.clips[0].transition_in == "cut"
    assert timeline.clips[0].primary_subject == "mountain lake"
    assert timeline.clips[0].subject_motion == "approaching"
    assert timeline.clips[0].action_progression == ["The hiker approaches the lake viewpoint"]
    assert timeline.clips[0].dynamics_confidence == 0.9
    assert timeline.clips[0].camera_motion == "tracking"
    assert timeline.clips[0].composition == "leading_lines"
    assert "Opening shot" in timeline.clips[0].continuity_note
    assert timeline.clips[1].transition_in == "crossfade"
    assert timeline.clips[1].transition_duration_seconds == 0.5
    assert timeline.proposed_duration_seconds == 5.5
    assert all(clip.decision == "pending" for clip in timeline.clips)
    assert timeline.timeline_approved is False


def test_compile_storyboard_requires_explicit_plan_approval(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)

    with pytest.raises(ValueError, match="approve the fully footage-mapped storyboard"):
        build_storyboard_timeline(tmp_path, plan_path=plan_path)


def test_save_storyboard_timeline_tracks_plan_and_editing_dependencies(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    approve_storyboard(tmp_path, plan_path=plan_path)
    timeline = build_storyboard_timeline(tmp_path)
    output = tmp_path / ".video-agent" / "optimized-timeline.json"

    save_storyboard_timeline(timeline, output)

    restored = ProjectDraftTimeline.model_validate_json(output.read_text(encoding="utf-8"))
    assert restored == timeline


def test_compile_storyboard_warns_but_allows_overlapping_source_intervals(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    plan = VideoProjectPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    second_shot = plan.scenes[0].shots[1]
    second_shot.footage_start_seconds = 0
    second_shot.footage_end_seconds = 8
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    register_artifact(tmp_path, plan_path)

    approve_storyboard(tmp_path, plan_path=plan_path)
    with pytest.warns(UserWarning, match="reuse overlapping source footage"):
        timeline = build_storyboard_timeline(tmp_path, plan_path=plan_path)

    assert len(timeline.clips) == 2

