import json
from pathlib import Path

import click
import pytest
from typer.testing import CliRunner

from video_edit_agent.cli import app
from video_edit_agent.models import ProjectAnalysis
from video_edit_agent.silence.schemas import ProjectSilenceAnalysis
from video_edit_agent.transcription.schemas import ProjectTranscription
from video_edit_agent.understanding.schemas import VideoUnderstanding

runner = CliRunner()


@pytest.mark.parametrize("output_option", [None, "--output", "-o"])
def test_analyze_command_writes_json(
    tmp_path: Path, monkeypatch, output_option: str | None
) -> None:
    from video_edit_agent import cli

    expected = {
        "project_path": str(tmp_path.resolve()),
        "video_count": 0,
        "total_duration_seconds": 0,
        "videos": [],
    }
    monkeypatch.setattr(
        cli.understanding,
        "analyze_folder",
        lambda folder: ProjectAnalysis(**expected),
    )
    output = (
        tmp_path / "custom" / "analysis.json"
        if output_option
        else tmp_path / ".video-agent" / "analysis.json"
    )
    args = ["analyze", str(tmp_path)]
    if output_option:
        args.extend([output_option, str(output)])
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected


def test_understand_command_writes_json(tmp_path: Path, monkeypatch) -> None:
    from video_edit_agent import cli

    expected = VideoUnderstanding(
        project_path=str(tmp_path.resolve()),
        video_count=1,
        scene_count=0,
        scenes=[],
        model="qwen2.5vl:7b",
    )
    monkeypatch.setattr(
        cli.understanding,
        "understand_folder",
        lambda *args, **kwargs: expected,
    )
    output = tmp_path / "custom" / "understanding.json"
    result = runner.invoke(app, ["understand", str(tmp_path), "--output", str(output)])
    assert result.exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected.model_dump(mode="json")


def test_transcribe_command_writes_json(tmp_path: Path, monkeypatch) -> None:
    from video_edit_agent import cli

    expected = ProjectTranscription(
        project_path=str(tmp_path.resolve()),
        video_count=1,
        transcribed_video_count=1,
        segment_count=1,
        model="tiny",
        videos=[],
    )
    monkeypatch.setattr(
        cli.understanding,
        "transcribe_folder",
        lambda *args, **kwargs: expected,
    )
    output = tmp_path / "custom" / "transcription.json"
    result = runner.invoke(app, ["transcribe", str(tmp_path), "--model", "tiny", "-o", str(output)])
    assert result.exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected.model_dump(mode="json")


def test_detect_silence_command_writes_json(tmp_path: Path, monkeypatch) -> None:
    from video_edit_agent import cli

    expected = ProjectSilenceAnalysis(
        project_path=str(tmp_path.resolve()),
        video_count=1,
        analyzed_video_count=1,
        noise_threshold_db=-35,
        min_silence_duration=1.5,
        speech_padding_seconds=0.25,
        videos=[],
    )
    monkeypatch.setattr(
        cli.understanding,
        "analyze_silence_folder",
        lambda *args, **kwargs: expected,
    )
    output = tmp_path / "custom" / "silence.json"
    result = runner.invoke(
        app, ["detect-silence", str(tmp_path), "--noise-db", "-35", "-o", str(output)]
    )
    assert result.exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected.model_dump(mode="json")


def test_cli_help_is_available() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "analyze" in result.output
    assert "understand" in result.output
    assert "transcribe" in result.output
    assert "detect-silence" in result.output
    assert "subtitles" in result.output
    assert "create-video" in result.output
    assert "story" in result.output
    assert "music" in result.output
    assert "render" in result.output
    assert "find-highlights" not in result.output
    assert "suggest-soundtrack" not in result.output
    assert "render-social-exports" not in result.output
    assert "validate-artifacts" in result.output


def test_propose_edits_command_writes_review_only_json(tmp_path: Path, monkeypatch) -> None:
    from video_edit_agent import cli
    from video_edit_agent.editing.schemas import (
        EditReviewItem,
        ProjectEditProposal,
        VideoEditProposal,
    )

    expected = ProjectEditProposal(
        project_path=str(tmp_path.resolve()),
        source_artifacts=[".video-agent/silence-analysis.json"],
        video_count=1,
        review_item_count=1,
        videos=[
            VideoEditProposal(
                path="clip.mp4",
                duration_seconds=10,
                review_items=[
                    EditReviewItem(
                        item_id="clip-review-001",
                        start_seconds=2,
                        end_seconds=4,
                        duration_seconds=2,
                        review_reason="Review the footage.",
                    )
                ],
            )
        ],
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        cli.understanding,
        "build_edit_proposal",
        lambda folder: expected,
    )
    output = tmp_path / "custom" / "proposal.json"
    result = runner.invoke(app, ["propose-edits", str(tmp_path), "-o", str(output)])
    monkeypatch.undo()

    assert result.exit_code == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved == expected.model_dump(mode="json")
    assert saved["approval_required"] is True
    assert saved["rendering_performed"] is False


def test_subtitles_command_writes_srt_path(tmp_path: Path, monkeypatch) -> None:
    from video_edit_agent import cli

    destination = tmp_path / "custom" / "captions.srt"
    monkeypatch.setattr(
        cli.understanding,
        "generate_subtitles",
        lambda *args, **kwargs: (destination, 7),
    )
    result = runner.invoke(app, ["subtitles", str(tmp_path), "--output", str(destination)])

    assert result.exit_code == 0, result.output
    assert "7 cue(s)" in result.output
    assert str(destination) in result.output


def test_cli_help_includes_propose_edits() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "propose-edits" in result.output
    assert "review-edits" in result.output
    assert "build-timeline" in result.output
    assert "render" in result.output
    assert "subtitles" in result.output


def test_render_help_includes_subtitle_style() -> None:
    result = runner.invoke(app, ["render", "video", "--help"])
    assert result.exit_code == 0
    assert "--subtitle-style" in click.unstyle(result.output)


def test_review_and_build_timeline_commands(tmp_path: Path) -> None:
    from video_edit_agent.editing.schemas import (
        EditReviewItem,
        ProjectEditProposal,
        VideoEditProposal,
    )

    artifact_dir = tmp_path / ".video-agent"
    artifact_dir.mkdir()
    proposal = ProjectEditProposal(
        project_path=str(tmp_path.resolve()),
        source_artifacts=[".video-agent/silence-analysis.json"],
        video_count=1,
        review_item_count=2,
        videos=[
            VideoEditProposal(
                path="clip.mp4",
                duration_seconds=10,
                review_items=[
                    EditReviewItem(
                        item_id="clip-review-001",
                        start_seconds=2,
                        end_seconds=4,
                        duration_seconds=2,
                        review_reason="Review first interval.",
                    ),
                    EditReviewItem(
                        item_id="clip-review-002",
                        start_seconds=6,
                        end_seconds=7,
                        duration_seconds=1,
                        review_reason="Review second interval.",
                    ),
                ],
            )
        ],
    )
    proposal_path = artifact_dir / "edit-proposal.json"
    proposal_path.write_text(proposal.model_dump_json(), encoding="utf-8")

    review = runner.invoke(
        app,
        [
            "review-edits",
            str(tmp_path),
            "--approve",
            "clip-review-001",
            "--reject",
            "clip-review-002",
        ],
    )
    assert review.exit_code == 0, review.output

    build = runner.invoke(app, ["build-timeline", str(tmp_path)])
    assert build.exit_code == 0, build.output
    saved = json.loads((artifact_dir / "timeline.json").read_text(encoding="utf-8"))
    assert saved["approved_cut_count"] == 1
    assert saved["retained_duration_seconds"] == 8
    assert saved["rendering_performed"] is False
    assert saved["videos"][0]["segments"] == [
        {"start_seconds": 0, "end_seconds": 2},
        {"start_seconds": 4, "end_seconds": 10},
    ]


def test_build_timeline_requires_all_review_items_decided(tmp_path: Path) -> None:
    from video_edit_agent.editing.schemas import (
        EditReviewItem,
        ProjectEditProposal,
        VideoEditProposal,
    )

    artifact_dir = tmp_path / ".video-agent"
    artifact_dir.mkdir()
    proposal = ProjectEditProposal(
        project_path=str(tmp_path.resolve()),
        source_artifacts=[],
        video_count=1,
        review_item_count=1,
        videos=[
            VideoEditProposal(
                path="clip.mp4",
                duration_seconds=10,
                review_items=[
                    EditReviewItem(
                        item_id="clip-review-001",
                        start_seconds=2,
                        end_seconds=4,
                        duration_seconds=2,
                        review_reason="Needs review.",
                    )
                ],
            )
        ],
    )
    (artifact_dir / "edit-proposal.json").write_text(proposal.model_dump_json(), encoding="utf-8")
    result = runner.invoke(app, ["build-timeline", str(tmp_path)])
    assert result.exit_code == 1
    assert "all review items must be approved or rejected" in result.output


@pytest.mark.parametrize("command", [["render", "video"], ["render-video"]])
def test_render_command_reports_verified_output(tmp_path: Path, monkeypatch, command) -> None:
    from video_edit_agent import cli
    from video_edit_agent.rendering import RenderReport

    expected = RenderReport(
        output_path=str(tmp_path / "rendered.mp4"),
        source_timeline=str(tmp_path / ".video-agent" / "timeline.json"),
        source_video_count=2,
        segment_count=4,
        expected_duration_seconds=18,
        actual_duration_seconds=18.02,
        width=1920,
        height=1080,
        video_codec="h264",
        audio_codec="aac",
    )
    captured = {}

    def fake_render(folder, **kwargs):
        captured["folder"] = folder
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        cli.rendering_core,
        "render_project",
        fake_render,
    )
    output = tmp_path / "custom.mp4"
    result = runner.invoke(
        app,
        [
            *command,
            str(tmp_path),
            "--output",
            str(output),
            "--overwrite",
            "--timeout",
            "120",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Render completed and verified" in result.output
    assert captured["output_path"] == output
    assert captured["overwrite"] is True
    assert captured["timeout_seconds"] == 120


def test_render_command_dry_run_reports_preflight(tmp_path: Path, monkeypatch) -> None:
    from video_edit_agent import cli

    captured = {}

    def fake_render(folder, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(
        cli.rendering_core,
        "render_project",
        fake_render,
    )
    result = runner.invoke(app, ["render-video", str(tmp_path), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert captured["dry_run"] is True
    assert "filter-graph preflight passed" in result.output
    assert "no MP4 was written" in result.output


def test_plan_command_writes_validated_editing_plan(tmp_path: Path, monkeypatch) -> None:
    from video_edit_agent import cli
    from video_edit_agent.editing.intent_schemas import EditingPlan

    expected = EditingPlan(
        source_prompt="Create a 60-second cinematic travel video",
        target_duration_seconds=60,
        style="cinematic",
        priorities=["scenic_landscapes", "memorable_moments"],
    )
    captured = {}

    def fake_parse(prompt: str, model: str | None = None):
        captured["prompt"] = prompt
        captured["model"] = model
        return expected

    monkeypatch.setattr(
        cli.planning_story,
        "parse_editing_intent",
        fake_parse,
    )
    output = tmp_path / "custom" / "editing-plan.json"
    result = runner.invoke(
        app,
        [
            "plan",
            str(tmp_path),
            "--prompt",
            "Create a 60-second cinematic travel video",
            "--model",
            "qwen2.5vl:3b",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved == expected.model_dump(mode="json")
    assert captured == {
        "prompt": "Create a 60-second cinematic travel video",
        "model": "qwen2.5vl:3b",
    }
    assert "No clips were selected" in result.output


def test_draft_edit_command_writes_selection_json(tmp_path: Path, monkeypatch) -> None:
    from video_edit_agent import cli
    from video_edit_agent.editing.draft_schemas import DraftClip, ProjectDraftSelection

    expected = ProjectDraftSelection(
        project_path=str(tmp_path.resolve()),
        source_artifacts=[".video-agent/editing-plan.json", ".video-agent/highlights.json"],
        target_duration_seconds=5,
        selected_duration_seconds=5,
        selected_clip_count=1,
        candidates_considered=1,
        clips=[
            DraftClip(
                clip_id="clip-1",
                video_path="travel.mp4",
                source_start_seconds=0,
                source_end_seconds=5,
                duration_seconds=5,
                highlight_score=90,
                relevance_score=80,
                combined_score=86.5,
                selected=True,
            )
        ],
    )
    monkeypatch.setattr(
        cli.planning_story,
        "build_draft_selection",
        lambda folder, plan_path=None: expected,
    )
    output = tmp_path / "custom" / "draft-selection.json"
    result = runner.invoke(
        app,
        ["draft-edit", str(tmp_path), "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text(encoding="utf-8")) == expected.model_dump(mode="json")
    assert "No existing timeline was changed" in result.output


def test_create_plan_command_writes_hierarchical_brief(tmp_path: Path, monkeypatch) -> None:
    from video_edit_agent import cli
    from video_edit_agent.editing.planning_schemas import VideoProjectPlan

    expected = VideoProjectPlan(
        project_path=str(tmp_path.resolve()),
        brief={
            "source_prompt": "Create a 90-second cinematic travel film",
            "target_duration_seconds": 90,
            "style": "cinematic",
            "aspect_ratio": "16:9",
            "resolution": "1080p",
            "pacing": "moderate",
            "creative_priorities": ["scenic_landscapes"],
        },
        editing_preferences={
            "source_prompt": "Create a 90-second cinematic travel film",
            "target_duration_seconds": 90,
            "style": "cinematic",
            "priorities": ["scenic_landscapes"],
        },
        story_treatment={
            "working_title": "Beyond the Mountains",
            "logline": "A journey into the Rockies.",
            "narrative_arc": ["Arrival", "Discovery"],
            "emotional_journey": "Anticipation becomes wonder.",
        },
        scenes=[
            {
                "scene_id": "scene-1",
                "sequence_index": 1,
                "title": "Arrival",
                "purpose": "Introduce the landscape.",
                "narrative_function": "hook",
                "target_duration_seconds": 90,
                "shots": [
                    {
                        "shot_id": "shot-1",
                        "scene_id": "scene-1",
                        "sequence_index": 1,
                        "purpose": "Reveal the mountains.",
                        "visual_description": "A wide mountain panorama.",
                        "target_duration_seconds": 45,
                    },
                    {
                        "shot_id": "shot-2",
                        "scene_id": "scene-1",
                        "sequence_index": 2,
                        "purpose": "Move toward the lake.",
                        "visual_description": "A slow approach to a turquoise lake.",
                        "target_duration_seconds": 45,
                    },
                ],
            }
        ],
        planning_stage="storyboard_ready",
    )
    destination = tmp_path / ".video-agent" / "video-project-plan.json"

    def fake_create(
        folder, prompt, *, model=None, output_path=None, music_analysis_path=None, story_path=None
    ):
        assert prompt == "Create a 90-second cinematic travel film"
        assert music_analysis_path is None
        return expected, destination

    monkeypatch.setattr(
        cli.planning_story,
        "create_video_project_plan",
        fake_create,
    )
    result = runner.invoke(
        app,
        [
            "create-plan",
            str(tmp_path),
            "--prompt",
            "Create a 90-second cinematic travel film",
            "--model",
            "qwen2.5vl:3b",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "story and storyboard plan created" in result.output
    assert "Beyond the Mountains" in result.output
    assert "storyboard_ready" in result.output


def test_suggest_footage_command_help():
    result = runner.invoke(app, ["suggest-footage", "--help"])
    assert result.exit_code == 0
    help_text = " ".join(click.unstyle(result.stdout).split())
    assert "--max-candidates" in help_text
    assert "never assign footage automatically" in help_text


def test_suggest_music_sync_command_help():
    result = runner.invoke(app, ["suggest-music-sync", "--help"])
    assert result.exit_code == 0
    help_text = " ".join(click.unstyle(result.stdout).split())
    assert "--music-analysis" in help_text
    assert "--tolerance" in help_text


def test_create_plan_command_exposes_music_analysis_option():
    result = runner.invoke(app, ["create-plan", "--help"])
    assert result.exit_code == 0
    help_text = " ".join(click.unstyle(result.output).split())
    assert "--music-analysis" in help_text


def test_create_plan_command_exposes_story_option():
    result = runner.invoke(app, ["create-plan", "--help"])
    assert result.exit_code == 0
    help_text = " ".join(click.unstyle(result.output).split())
    assert "--story" in help_text


def _music_sync_cli_inputs(tmp_path: Path) -> tuple[Path, Path]:
    from video_edit_agent.editing.timeline_schemas import DraftTimelineClip, ProjectDraftTimeline
    from video_edit_agent.music.sync import MusicSyncBoundary, MusicSyncSuggestions

    artifacts = tmp_path / ".video-agent"
    artifacts.mkdir(exist_ok=True)
    timeline = artifacts / "optimized-timeline.json"
    clips = [
        DraftTimelineClip(
            clip_id="shot-001",
            sequence_index=1,
            video_path="a.mp4",
            source_start_seconds=1.0,
            source_end_seconds=5.0,
            duration_seconds=4.0,
            combined_score=90,
            pacing_note="Opening shot",
        ),
        DraftTimelineClip(
            clip_id="shot-002",
            sequence_index=2,
            video_path="b.mp4",
            source_start_seconds=2.0,
            source_end_seconds=6.0,
            duration_seconds=4.0,
            combined_score=90,
            pacing_note="Following shot",
        ),
    ]
    timeline.write_text(
        ProjectDraftTimeline(
            project_path=str(tmp_path.resolve()),
            style="cinematic",
            pacing="moderate",
            target_duration_seconds=8,
            proposed_duration_seconds=8,
            proposed_clip_count=2,
            clips=clips,
        ).model_dump_json(),
        encoding="utf-8",
    )
    suggestions = artifacts / "music-sync-suggestions.json"
    report = MusicSyncSuggestions(
        project_path=str(tmp_path.resolve()),
        timeline_path=timeline.relative_to(tmp_path).as_posix(),
        music_analysis_path=".video-agent/music-analysis.json",
        music_path=str(tmp_path / "track.mp3"),
        beat_confidence=0.9,
        tolerance_seconds=0.18,
        boundary_count=1,
        aligned_boundary_count=1,
        boundaries=[
            MusicSyncBoundary(
                previous_clip_id="shot-001",
                next_clip_id="shot-002",
                boundary_seconds=4.0,
                transition="cut",
                transition_duration_seconds=0,
                recommended_anchor="beat",
                recommended_boundary_seconds=4.2,
                adjustment_seconds=0.2,
            )
        ],
    )
    suggestions.write_text(report.model_dump_json(), encoding="utf-8")
    return timeline, suggestions


def test_review_music_sync_command_help() -> None:
    result = runner.invoke(app, ["review-music-sync", "--help"], terminal_width=120)
    assert result.exit_code == 0, result.output
    help_text = "".join(click.unstyle(result.output).split())
    for option in (
        "--accept",
        "--reject",
        "--timeline",
        "--suggestions",
        "--minimum-clip-duration",
    ):
        assert option in help_text


def test_review_music_sync_command_accepts_and_persists_selected_boundary(
    tmp_path: Path, monkeypatch
) -> None:
    from video_edit_agent import cli
    from video_edit_agent.music.refinement import (
        MusicSyncReviewDecision,
        MusicSyncReviewResult,
    )

    _music_sync_cli_inputs(tmp_path)
    captured = {}

    def fake_review(folder, **kwargs):
        captured.update(kwargs)
        return MusicSyncReviewResult(
            source_timeline_path=".video-agent/optimized-timeline.json",
            suggestions_path=".video-agent/music-sync-suggestions.json",
            revision_path=".video-agent/music-sync-revisions/revision-001.json",
            decisions=[
                MusicSyncReviewDecision(boundary_id="shot-001->shot-002", status="accepted")
            ],
        )

    monkeypatch.setattr(
        cli.music,
        "apply_music_sync_review",
        fake_review,
    )
    result = runner.invoke(
        app, ["review-music-sync", str(tmp_path), "--accept", "shot-001->shot-002"]
    )

    assert result.exit_code == 0, result.output
    assert captured["accept_ids"] == {"shot-001->shot-002"}
    assert captured["reject_ids"] == set()
    assert "review-timeline" in result.output
    assert "No video was rendered" in result.output


def test_review_music_sync_command_rejects_overlapping_accept_and_reject_ids(
    tmp_path: Path, monkeypatch
) -> None:
    from video_edit_agent import cli

    called = False

    def fake_review(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(
        cli.music,
        "apply_music_sync_review",
        fake_review,
    )
    result = runner.invoke(
        app,
        [
            "review-music-sync",
            str(tmp_path),
            "--accept",
            "shot-001->shot-002",
            "--reject",
            "shot-001->shot-002",
        ],
    )

    assert result.exit_code != 0
    assert called is False
    assert "both accepted and rejected" in result.output


def test_review_music_sync_command_interactively_prompts_for_each_pending_boundary(
    tmp_path: Path, monkeypatch
) -> None:
    from video_edit_agent import cli
    from video_edit_agent.music.refinement import (
        MusicSyncReviewDecision,
        MusicSyncReviewResult,
    )

    _music_sync_cli_inputs(tmp_path)
    captured = {}

    def fake_review(folder, **kwargs):
        captured.update(kwargs)
        return MusicSyncReviewResult(
            source_timeline_path=".video-agent/optimized-timeline.json",
            suggestions_path=".video-agent/music-sync-suggestions.json",
            revision_path=".video-agent/music-sync-revisions/revision-001.json",
            decisions=[
                MusicSyncReviewDecision(boundary_id="shot-001->shot-002", status="accepted")
            ],
        )

    monkeypatch.setattr(
        cli.music,
        "apply_music_sync_review",
        fake_review,
    )
    result = runner.invoke(app, ["review-music-sync", str(tmp_path)], input="a\n")

    assert result.exit_code == 0, result.output
    assert captured["accept_ids"] == {"shot-001->shot-002"}
    assert captured["reject_ids"] == set()
    assert "Current boundary: 4.000s" in result.output
    assert "Suggested: 4.200s" in result.output


def test_suggest_soundtrack_command_writes_brief(tmp_path: Path, monkeypatch) -> None:
    from video_edit_agent import cli
    from video_edit_agent.soundtrack.schemas import SoundtrackBrief, SoundtrackEnergySegment

    expected = SoundtrackBrief(
        project_path=str(tmp_path.resolve()),
        source_video_plan=".video-agent/video-project-plan.json",
        target_duration_seconds=12,
        video_style="cinematic",
        narrative_summary="A quiet journey through the mountains.",
        mood=["wonder", "calm"],
        genre_directions=["ambient", "cinematic"],
        instrumentation=["piano", "strings"],
        vocals="instrumental_preferred",
        tempo_min_bpm=75,
        tempo_max_bpm=120,
        energy_arc=[
            SoundtrackEnergySegment(
                scene_id="scene-1",
                scene_title="Arrival",
                start_seconds=0,
                end_seconds=12,
                energy="low",
            )
        ],
        search_terms=["wonder", "ambient", "instrumental"],
        generation_prompt="Create a calm instrumental soundtrack.",
    )
    destination = tmp_path / ".video-agent" / "soundtrack-brief.json"
    monkeypatch.setattr(
        cli.music,
        "create_soundtrack_brief",
        lambda folder, *, plan_path=None, output_path=None: (expected, destination),
    )

    result = runner.invoke(app, ["suggest-soundtrack", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "Soundtrack brief created" in result.output
    assert "No track was downloaded or generated" in result.output


def test_analyze_selected_soundtrack_command_invokes_provenance_workflow(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from video_edit_agent import cli

    expected = (
        SimpleNamespace(bpm=92.0, duration_seconds=60.0, beat_confidence=0.8),
        SimpleNamespace(candidate_id="track-1", provider_id="test-provider"),
        tmp_path / ".video-agent" / "music-analysis.json",
        tmp_path / ".video-agent" / "selected-soundtrack-analysis.json",
    )
    captured = {}

    def fake_analyze(folder, **kwargs):
        captured["folder"] = folder
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(
        cli.music,
        "analyze_selected_soundtrack",
        fake_analyze,
    )
    result = runner.invoke(
        app,
        [
            "analyze-selected-soundtrack",
            str(tmp_path),
            "--music",
            ".video-agent/imported-track.mp3",
            "--output",
            ".video-agent/custom-analysis.json",
        ],
    )

    assert result.exit_code == 0
    assert captured["folder"] == tmp_path
    assert captured["music_path"] == Path(".video-agent/imported-track.mp3")
    assert captured["analysis_output_path"] == Path(".video-agent/custom-analysis.json")
    assert "no timeline approval or rendering occurred" in " ".join(result.output.split())


@pytest.mark.parametrize(
    ("group", "expected_commands"),
    [
        ("story", ["find-highlights", "create-plan", "propose-story", "review-timeline"]),
        ("music", ["suggest-soundtrack", "analyze-music", "review-music-sync"]),
        ("render", ["video", "render-social-exports", "critique-draft"]),
    ],
)
def test_cli_subcommand_groups_have_focused_help(group: str, expected_commands: list[str]) -> None:
    result = runner.invoke(app, [group, "--help"])
    assert result.exit_code == 0, result.output
    for command in expected_commands:
        assert command in result.output


def test_legacy_advanced_commands_remain_available_as_hidden_aliases() -> None:
    result = runner.invoke(app, ["find-highlights", "--help"])
    assert result.exit_code == 0
    assert "Folder containing video files" in result.output

    render_result = runner.invoke(app, ["render-video", "--help"])
    assert render_result.exit_code == 0
    assert "--overwrite" in click.unstyle(render_result.output)
