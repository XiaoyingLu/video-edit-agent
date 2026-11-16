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
