import pytest
from typer.testing import CliRunner

from travel_video_agent.cli import app
from travel_video_agent.editing.draft_schemas import DraftClip, ProjectDraftSelection
from travel_video_agent.editing.intent_schemas import EditingPlan
from travel_video_agent.editing.timeline_pipeline import (
    apply_timeline_review,
    build_draft_timeline,
)
from travel_video_agent.editing.timeline_schemas import (
    DraftTimelineClip,
    ProjectDraftTimeline,
)

runner = CliRunner()


def _write_inputs(root, style="vlog", transition="crossfade"):
    artifact = root / ".video-agent"
    artifact.mkdir(parents=True, exist_ok=True)
    plan = EditingPlan(
        source_prompt="Create a travel video",
        target_duration_seconds=12,
        style=style,
        pacing="moderate",
        priorities=(
            ["chronological_story"] if style != "highlight_reel" else ["memorable_moments"]
        ),
        transitions={
            "default": transition,
            "duration_seconds": 0.8 if transition != "cut" else 0,
        },
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
    assert timeline.approved_duration_seconds == 0
    assert timeline.timeline_approved is False
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
    assert approved.timeline_approved is True
    assert approved.approved_duration_seconds == 4
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
    assert reviewed.approved_duration_seconds == 0


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
    assert "approve-storyboard" in result.output
    assert "build-storyboard-timeline" in result.output


def test_cinematic_timeline_builds_establishing_action_detail_and_scenic_arc(tmp_path):
    from travel_video_agent.editing.draft_schemas import DraftClip

    _write_inputs(tmp_path, style="highlight_reel", transition="cut")
    artifact = tmp_path / ".video-agent"
    plan = EditingPlan(
        source_prompt="Create a cinematic travel film",
        target_duration_seconds=24,
        style="cinematic",
        pacing="moderate",
        priorities=["scenic_landscapes"],
        transitions={"default": "cut", "duration_seconds": 0},
    )
    (artifact / "editing-plan.json").write_text(plan.model_dump_json(), encoding="utf-8")
    clips = [
        DraftClip(
            clip_id="action",
            video_path="trip.mp4",
            source_start_seconds=0,
            source_end_seconds=4,
            duration_seconds=4,
            highlight_score=99,
            relevance_score=90,
            combined_score=99,
            visual_role="action",
            shot_scale="medium",
            selected=True,
        ),
        DraftClip(
            clip_id="establishing",
            video_path="trip.mp4",
            source_start_seconds=4,
            source_end_seconds=10,
            duration_seconds=6,
            highlight_score=70,
            relevance_score=70,
            combined_score=70,
            visual_role="establishing",
            shot_scale="wide",
            selected=True,
        ),
        DraftClip(
            clip_id="detail",
            video_path="trip.mp4",
            source_start_seconds=10,
            source_end_seconds=13,
            duration_seconds=3,
            highlight_score=90,
            relevance_score=90,
            combined_score=90,
            visual_role="detail",
            shot_scale="close_up",
            selected=True,
        ),
        DraftClip(
            clip_id="scenic",
            video_path="trip.mp4",
            source_start_seconds=13,
            source_end_seconds=19,
            duration_seconds=6,
            highlight_score=65,
            relevance_score=65,
            combined_score=65,
            visual_role="scenic_breath",
            shot_scale="wide",
            mood="serene",
            selected=True,
        ),
    ]
    selection = ProjectDraftSelection(
        project_path=str(tmp_path.resolve()),
        source_artifacts=[".video-agent/editing-plan.json"],
        target_duration_seconds=24,
        selected_duration_seconds=19,
        selected_clip_count=4,
        candidates_considered=4,
        clips=clips,
    )
    (artifact / "draft-selection.json").write_text(selection.model_dump_json(), encoding="utf-8")

    timeline = build_draft_timeline(tmp_path)

    assert [clip.clip_id for clip in timeline.clips] == [
        "establishing",
        "action",
        "detail",
        "scenic",
    ]
    assert timeline.clips[0].sequence_reason.startswith("Opens with a wide establishing")
    assert "visual-scale contrast" in timeline.clips[2].sequence_reason
    assert timeline.clips[-1].sequence_reason.startswith("Closes on a scenic")
    assert timeline.clips[1].pacing_target_seconds == 3.0
    assert timeline.clips[-1].pacing_target_seconds == 4.5
    assert timeline.clips[-1].visual_role == "scenic_breath"


def test_chronological_story_keeps_source_order_despite_scene_roles(tmp_path):
    from travel_video_agent.editing.draft_schemas import DraftClip

    _write_inputs(tmp_path, style="vlog", transition="cut")
    artifact = tmp_path / ".video-agent"
    clips = [
        DraftClip(
            clip_id="later-establishing",
            video_path="trip.mp4",
            source_start_seconds=8,
            source_end_seconds=12,
            duration_seconds=4,
            highlight_score=90,
            relevance_score=90,
            combined_score=90,
            visual_role="establishing",
            selected=True,
        ),
        DraftClip(
            clip_id="earlier-action",
            video_path="trip.mp4",
            source_start_seconds=1,
            source_end_seconds=5,
            duration_seconds=4,
            highlight_score=80,
            relevance_score=80,
            combined_score=80,
            visual_role="action",
            selected=True,
        ),
    ]
    selection = ProjectDraftSelection(
        project_path=str(tmp_path.resolve()),
        source_artifacts=[".video-agent/editing-plan.json"],
        target_duration_seconds=12,
        selected_duration_seconds=8,
        selected_clip_count=2,
        candidates_considered=2,
        clips=clips,
    )
    (artifact / "draft-selection.json").write_text(selection.model_dump_json(), encoding="utf-8")

    timeline = build_draft_timeline(tmp_path)

    assert [clip.clip_id for clip in timeline.clips] == ["earlier-action", "later-establishing"]


def test_role_aware_pacing_uses_different_targets_for_action_and_scenic_shots():
    from travel_video_agent.editing.timeline_pipeline import _pacing_guidance

    action = _pacing_guidance(3.0, "moderate", "action")
    scenic = _pacing_guidance(4.5, "moderate", "scenic_breath")

    assert action[0] == "within"
    assert scenic[0] == "within"
    assert action[2] == 3.0
    assert scenic[2] == 4.5


def test_cinematic_timeline_preserves_ranked_order_without_scene_metadata(tmp_path):
    _write_inputs(tmp_path, style="highlight_reel", transition="cut")
    artifact = tmp_path / ".video-agent"
    plan = EditingPlan(
        source_prompt="Create a cinematic travel video",
        target_duration_seconds=12,
        style="cinematic",
        priorities=["memorable_moments"],
        transitions={"default": "cut", "duration_seconds": 0},
    )
    (artifact / "editing-plan.json").write_text(plan.model_dump_json(), encoding="utf-8")

    timeline = build_draft_timeline(tmp_path)

    assert [clip.clip_id for clip in timeline.clips] == ["later", "earlier"]
    assert all(clip.visual_role == "unknown" for clip in timeline.clips)
    assert "strongest available shot" in timeline.clips[0].sequence_reason
    assert "metadata is uncertain" in timeline.clips[1].sequence_reason



def test_music_analysis_guides_pacing_without_changing_source_intervals(tmp_path):
    import json

    from travel_video_agent.editing.draft_schemas import DraftClip
    from travel_video_agent.music.schemas import MusicAnalysis, MusicSection

    _write_inputs(tmp_path, style="vlog", transition="cut")
    artifact = tmp_path / ".video-agent"
    selection = ProjectDraftSelection.model_validate_json(
        (artifact / "draft-selection.json").read_text(encoding="utf-8")
    )
    selection.clips[0].source_start_seconds = 0
    selection.clips[0].source_end_seconds = 4
    selection.clips[0].duration_seconds = 4
    selection.clips[1].source_start_seconds = 4
    selection.clips[1].source_end_seconds = 8
    selection.clips[1].duration_seconds = 4
    (artifact / "draft-selection.json").write_text(selection.model_dump_json(), encoding="utf-8")
    soundtrack = tmp_path / "score.mp3"
    soundtrack.write_bytes(b"test soundtrack")
    analysis = MusicAnalysis(
        project_path=str(tmp_path.resolve()),
        music_path=str(soundtrack),
        duration_seconds=30,
        sample_rate=22050,
        beat_confidence=0.5,
        sections=[
            MusicSection(start_seconds=0, end_seconds=4, energy="low", energy_rms=0.1),
            MusicSection(start_seconds=4, end_seconds=30, energy="high", energy_rms=0.8),
        ],
    )
    analysis_path = artifact / "music-analysis.json"
    analysis_path.write_text(analysis.model_dump_json(), encoding="utf-8")

    timeline = build_draft_timeline(tmp_path, music_analysis_path=analysis_path)

    assert timeline.source_music_analysis == ".video-agent/music-analysis.json"
    assert [clip.music_energy for clip in timeline.clips] == ["low", "high"]
    assert timeline.clips[0].pacing_target_seconds > timeline.clips[1].pacing_target_seconds
    assert "breathing room" in timeline.clips[0].music_pacing_note
    assert "tighter shot" in timeline.clips[1].music_pacing_note
    assert [(clip.source_start_seconds, clip.source_end_seconds) for clip in timeline.clips] == [
        (1, 5), (8, 12)
    ]


def test_music_analysis_rejects_foreign_project(tmp_path):
    from travel_video_agent.music.schemas import MusicAnalysis

    _write_inputs(tmp_path)
    analysis_path = tmp_path / "music-analysis.json"
    analysis_path.write_text(
        MusicAnalysis(
            project_path=str(tmp_path / "other"),
            music_path=str(tmp_path / "score.mp3"),
            duration_seconds=20,
            sample_rate=22050,
            beat_confidence=0.2,
        ).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different project"):
        build_draft_timeline(tmp_path, music_analysis_path=analysis_path)
