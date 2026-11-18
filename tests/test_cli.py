import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from travel_video_agent.cli import app
from travel_video_agent.models import ProjectAnalysis
from travel_video_agent.silence.schemas import ProjectSilenceAnalysis
from travel_video_agent.transcription.schemas import ProjectTranscription
from travel_video_agent.understanding.schemas import VideoUnderstanding

runner = CliRunner()


@pytest.mark.parametrize("output_option", [None, "--output", "-o"])
def test_analyze_command_writes_json(
    tmp_path: Path, monkeypatch, output_option: str | None
) -> None:
    from travel_video_agent import cli

    expected = {
        "project_path": str(tmp_path.resolve()),
        "video_count": 0,
        "total_duration_seconds": 0,
        "videos": [],
    }
    monkeypatch.setattr(cli, "analyze_folder", lambda folder: ProjectAnalysis(**expected))
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
    from travel_video_agent import cli

    expected = VideoUnderstanding(
        project_path=str(tmp_path.resolve()),
        video_count=1,
        scene_count=0,
        scenes=[],
        model="qwen2.5vl:7b",
    )
    monkeypatch.setattr(cli, "understand_folder", lambda *args, **kwargs: expected)
    output = tmp_path / "custom" / "understanding.json"
    result = runner.invoke(app, ["understand", str(tmp_path), "--output", str(output)])
    assert result.exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected.model_dump(mode="json")


def test_transcribe_command_writes_json(tmp_path: Path, monkeypatch) -> None:
    from travel_video_agent import cli

    expected = ProjectTranscription(
        project_path=str(tmp_path.resolve()),
        video_count=1,
        transcribed_video_count=1,
        segment_count=1,
        model="tiny",
        videos=[],
    )
    monkeypatch.setattr(cli, "transcribe_folder", lambda *args, **kwargs: expected)
    output = tmp_path / "custom" / "transcription.json"
    result = runner.invoke(app, ["transcribe", str(tmp_path), "--model", "tiny", "-o", str(output)])
    assert result.exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected.model_dump(mode="json")


def test_detect_silence_command_writes_json(tmp_path: Path, monkeypatch) -> None:
    from travel_video_agent import cli

    expected = ProjectSilenceAnalysis(
        project_path=str(tmp_path.resolve()),
        video_count=1,
        analyzed_video_count=1,
        noise_threshold_db=-35,
        min_silence_duration=1.5,
        speech_padding_seconds=0.25,
        videos=[],
    )
    monkeypatch.setattr(cli, "analyze_silence_folder", lambda *args, **kwargs: expected)
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
    assert "find-highlights" in result.output
    assert "plan" in result.output
    assert "draft-edit" in result.output
    assert "create-video" in result.output
    assert "create-plan" in result.output
    assert "validate-artifacts" in result.output


def test_propose_edits_command_writes_review_only_json(tmp_path: Path, monkeypatch) -> None:
    from travel_video_agent import cli
    from travel_video_agent.editing.schemas import (
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
    monkeypatch.setattr(cli, "build_edit_proposal", lambda folder: expected)
    output = tmp_path / "custom" / "proposal.json"
    result = runner.invoke(app, ["propose-edits", str(tmp_path), "-o", str(output)])
    monkeypatch.undo()

    assert result.exit_code == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved == expected.model_dump(mode="json")
    assert saved["approval_required"] is True
    assert saved["rendering_performed"] is False


def test_subtitles_command_writes_srt_path(tmp_path: Path, monkeypatch) -> None:
    from travel_video_agent import cli

    destination = tmp_path / "custom" / "captions.srt"
    monkeypatch.setattr(cli, "generate_subtitles", lambda *args, **kwargs: (destination, 7))
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


def test_review_and_build_timeline_commands(tmp_path: Path) -> None:
    from travel_video_agent.editing.schemas import (
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
    from travel_video_agent.editing.schemas import (
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


def test_render_command_reports_verified_output(tmp_path: Path, monkeypatch) -> None:
    from travel_video_agent import cli
    from travel_video_agent.rendering import RenderReport

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

    monkeypatch.setattr(cli, "render_project", fake_render)
    output = tmp_path / "custom.mp4"
    result = runner.invoke(
        app,
        [
            "render",
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
    from travel_video_agent import cli

    captured = {}

    def fake_render(folder, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(cli, "render_project", fake_render)
    result = runner.invoke(app, ["render", str(tmp_path), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert captured["dry_run"] is True
    assert "filter-graph preflight passed" in result.output
    assert "no MP4 was written" in result.output


def test_plan_command_writes_validated_editing_plan(tmp_path: Path, monkeypatch) -> None:
    from travel_video_agent import cli
    from travel_video_agent.editing.intent_schemas import EditingPlan

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

    monkeypatch.setattr(cli, "parse_editing_intent", fake_parse)
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
    from travel_video_agent import cli
    from travel_video_agent.editing.draft_schemas import DraftClip, ProjectDraftSelection

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
    monkeypatch.setattr(cli, "build_draft_selection", lambda folder, plan_path=None: expected)
    output = tmp_path / "custom" / "draft-selection.json"
    result = runner.invoke(
        app,
        ["draft-edit", str(tmp_path), "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text(encoding="utf-8")) == expected.model_dump(mode="json")
    assert "No existing timeline was changed" in result.output


def test_create_plan_command_writes_hierarchical_brief(tmp_path: Path, monkeypatch) -> None:
    from travel_video_agent import cli
    from travel_video_agent.editing.planning_schemas import VideoProjectPlan

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

    def fake_create(folder, prompt, *, model=None, output_path=None):
        assert prompt == "Create a 90-second cinematic travel film"
        return expected, destination

    monkeypatch.setattr(cli, "create_video_project_plan", fake_create)
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
    assert "--max-candidates" in result.stdout
    assert "never assign footage automatically" in result.stdout
