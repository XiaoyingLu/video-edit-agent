import pytest

from travel_video_agent.editing.draft_schemas import DraftClip, ProjectDraftSelection
from travel_video_agent.editing.intent_schemas import EditingPlan
from travel_video_agent.editing.timeline_pipeline import (
    apply_timeline_review,
    build_draft_timeline,
)
from typer.testing import CliRunner\n\nfrom travel_video_agent.cli import app\nfrom travel_video_agent.editing.timeline_schemas import (\n    DraftTimelineClip,\n    ProjectDraftTimeline,\n)

runner = CliRunner()


def _write_inputs(root, style="vlog", transition="crossfade"):
    artifact = root / ".video-agent"
    artifact.mkdir(parents=True, exist_ok=True)
    plan = EditingPlan(
        source_prompt="Create a travel video",
        target_duration_seconds=12,
        style=style,
        pacing="moderate",
        priorities=(["chronological_story"] if style != "highlight_reel" else ["memorable_moments"]),
        transitions={"default": transition, "duration_seconds": 0.8 if transition != "cut" else 0},
    )
    selection = ProjectDraftSelection(
        project_path=str(root.resolve()),
        source_artifacts=[".video-agent/editing-plan.json", ".video-agent/highlights.json"],
        target_duration_seconds=12,
        selected_duration_seconds=8,
        selected_clip_count=2,
        candidates_considered=2,
        clips=[
            DraftClip(
                clip_id="later",
                video_path="trip.mp4",
                source_start_seconds=8,
                source_end_seconds=12,
                duration_seconds=4,
                highlight_score=90,
                relevance_score=80,
                combined_score=86.5,
                scene_summary="Mountain viewpoint",
                selected=True,
            ),
            DraftClip(
                clip_id="earlier",
                video_path="trip.mp4",
                source_start_seconds=1,
                source_end_seconds=5,
                duration_seconds=4,
                highlight_score=80,
                relevance_score=70,
                combined_score=76.5,
                scene_summary="Walking through town",
                selected=True,
            ),
        ],
    )
    (artifact / "editing-plan.json").write_text(plan.model_dump_json(), encoding="utf-8")
    (artifact / "draft-selection.json").write_text(selection.model_dump_json(), encoding="utf-8")


def test_build_draft_timeline_orders_chronologically_and_proposes_transitions(tmp_path):
    _write_inputs(tmp_path)
    timeline = build_draft_timeline(tmp_path)

    assert [clip.clip_id for clip in timeline.clips] == ["earlier", "later"]
    assert [clip.sequence_index for clip in timeline.clips] == [1, 2]
    assert timeline.clips[0].transition_in == "cut"
    assert timeline.clips[0].transition_duration_seconds == 0
    assert timeline.clips[1].transition_in == "crossfade"
    assert timeline.clips[1].transition_duration_seconds == 0.8
    assert timeline.proposed_duration_seconds == 7.2
    assert timeline.approved_duration_seconds == 0\n    assert timeline.timeline_approved is False
    assert timeline.approval_required is True
    assert timeline.rendering_performed is False


def test_highlight_reel_preserves_ranked_order(tmp_path):
    _write_inputs(tmp_path, style="highlight_reel", transition="cut")
    timeline = build_draft_timeline(tmp_path)
    assert [clip.clip_id for clip in timeline.clips] == ["later", "earlier"]


def test_timeline_review_requires_all_clips_decided(tmp_path):
    _write_inputs(tmp_path)
    timeline = build_draft_timeline(tmp_path)

    partially_reviewed = apply_timeline_review(timeline, {"earlier"}, set())
    assert partially_reviewed.timeline_approved is False
    assert sum(clip.decision == "pending" for clip in partially_reviewed.clips) == 1

    approved = apply_timeline_review(partially_reviewed, set(), {"later"})
    assert approved.timeline_approved is True\n    assert approved.approved_duration_seconds == 4
    assert [clip.decision for clip in approved.clips] == ["approved", "rejected"]


def test_timeline_review_rejects_unknown_or_conflicting_ids(tmp_path):
    _write_inputs(tmp_path)
    timeline = build_draft_timeline(tmp_path)

    with pytest.raises(ValueError, match="both"):
        apply_timeline_review(timeline, {"earlier"}, {"earlier"})
    with pytest.raises(ValueError, match="unknown"):
        apply_timeline_review(timeline, {"missing"}, set())


def test_timeline_schema_rejects_non_cut_transition_on_first_clip():
    with pytest.raises(ValueError, match="first clip"):
        ProjectDraftTimeline(
            project_path="/tmp/project",
            style="cinematic",
            pacing="slow",
            target_duration_seconds=5,
            proposed_duration_seconds=2,
            proposed_clip_count=1,
            clips=[
                DraftTimelineClip(
                    clip_id="clip-1",
                    sequence_index=1,
                    video_path="trip.mp4",
                    source_start_seconds=0,
                    source_end_seconds=2,
                    duration_seconds=2,
                    combined_score=90,
                    transition_in="crossfade",
                    transition_duration_seconds=0.2,
                    pacing_fit="short",
                    pacing_note="Short shot.",
                )
            ],
        )


def test_build_draft_timeline_requires_selection_and_plan(tmp_path):
    with pytest.raises(FileNotFoundError, match="draft-selection.json"):
        build_draft_timeline(tmp_path)


def test_timeline_review_requires_at_least_one_approved_clip(tmp_path):
    _write_inputs(tmp_path)
    timeline = build_draft_timeline(tmp_path)
    reviewed = apply_timeline_review(timeline, set(), {"earlier", "later"})
    assert reviewed.timeline_approved is False
    assert all(clip.decision == "rejected" for clip in reviewed.clips)



def test_optimize_and_review_timeline_cli_commands(tmp_path):
    _write_inputs(tmp_path)
    optimized = runner.invoke(app, ["optimize-timeline", str(tmp_path)])
    assert optimized.exit_code == 0, optimized.output
    artifact = tmp_path / ".video-agent" / "optimized-timeline.json"
    assert artifact.is_file()

    reviewed = runner.invoke(
        app,
        [
            "review-timeline",
            str(tmp_path),
            "--approve",
            "earlier",
            "--reject",
            "later",
        ],
    )
    assert reviewed.exit_code == 0, reviewed.output
    saved = ProjectDraftTimeline.model_validate_json(artifact.read_text(encoding="utf-8"))
    assert saved.timeline_approved is True
    assert [clip.decision for clip in saved.clips] == ["approved", "rejected"]


def test_cli_help_includes_timeline_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "optimize-timeline" in result.output
    assert "review-timeline" in result.output
