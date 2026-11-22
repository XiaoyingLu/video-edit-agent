import json
import shutil
import subprocess
from pathlib import Path

import click
import pytest
from typer.testing import CliRunner

from video_edit_agent.artifact_tracking import register_artifact
from video_edit_agent.cli import app
from video_edit_agent.editing.timeline_schemas import (
    DraftTimelineClip,
    ProjectDraftTimeline,
)
from video_edit_agent.rendering import RenderError, RenderReport, render_approved_shot_timeline
from video_edit_agent.rendering.shot_renderer import (
    _build_shot_render_command,
    _resolve_output_dimensions,
)

runner = CliRunner()


def _clip(
    clip_id: str,
    index: int,
    start: float,
    end: float,
    *,
    decision: str = "approved",
    transition: str = "cut",
    transition_duration: float = 0,
) -> DraftTimelineClip:
    return DraftTimelineClip(
        clip_id=clip_id,
        sequence_index=index,
        video_path="trip.mp4",
        source_start_seconds=start,
        source_end_seconds=end,
        duration_seconds=end - start,
        combined_score=90,
        transition_in=transition,
        transition_duration_seconds=transition_duration,
        pacing_fit="within",
        pacing_note="Within pacing range.",
        decision=decision,
    )


def _timeline(root: Path, *, approved: bool = True) -> ProjectDraftTimeline:
    clips = [
        _clip("one", 1, 0, 3, transition="cut"),
        _clip("two", 2, 5, 8, transition="crossfade", transition_duration=0.5),
    ]
    return ProjectDraftTimeline(
        project_path=str(root.resolve()),
        style="cinematic",
        pacing="moderate",
        target_duration_seconds=10,
        proposed_duration_seconds=5.5,
        approved_duration_seconds=5.5 if approved else 3,
        proposed_clip_count=2,
        clips=(
            clips
            if approved
            else [
                clips[0],
                _clip(
                    "two",
                    2,
                    5,
                    8,
                    decision="pending",
                    transition="crossfade",
                    transition_duration=0.5,
                ),
            ]
        ),
        timeline_approved=approved,
    )


def test_output_dimensions_follow_project_plan_and_require_fresh_artifact(tmp_path):
    artifact = tmp_path / ".video-agent"
    artifact.mkdir()
    plan_path = artifact / "video-project-plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "project_path": str(tmp_path.resolve()),
                "brief": {
                    "source_prompt": "Create a vertical video",
                    "target_duration_seconds": 10,
                    "style": "cinematic",
                    "aspect_ratio": "9:16",
                    "resolution": "1080p",
                    "pacing": "moderate",
                },
                "editing_preferences": {
                    "source_prompt": "Create a vertical video",
                    "target_duration_seconds": 10,
                    "style": "cinematic",
                    "aspect_ratio": "9:16",
                    "resolution": "1080p",
                    "pacing": "moderate",
                },
            }
        ),
        encoding="utf-8",
    )
    register_artifact(tmp_path, plan_path)

    assert _resolve_output_dimensions(tmp_path, 1920, 1080) == (1080, 1920)


def test_shot_renderer_builds_crossfade_and_audio_crossfade_graph(tmp_path):
    command = _build_shot_render_command(
        "ffmpeg",
        [
            (tmp_path / "one.mp4", 0, 3, True, "cut", 0),
            (tmp_path / "two.mp4", 5, 8, True, "crossfade", 0.5),
        ],
        tmp_path / "out.mp4",
        1920,
        1080,
    )
    graph = command[command.index("-filter_complex") + 1]
    assert "xfade=transition=fade:duration=0.500000:offset=2.500000" in graph
    assert "acrossfade=d=0.500000" in graph
    assert "[v0][a0][v1][a1]concat" not in graph


def test_shot_renderer_supports_crop_and_pad_fit_modes(tmp_path):
    crop = _build_shot_render_command(
        "ffmpeg",
        [(tmp_path / "one.mp4", 0, 2, False, "cut", 0)],
        tmp_path / "crop.mp4",
        1080,
        1920,
        fit_mode="crop",
    )
    crop_graph = crop[crop.index("-filter_complex") + 1]
    assert "force_original_aspect_ratio=increase,crop=1080:1920" in crop_graph
    pad = _build_shot_render_command(
        "ffmpeg",
        [(tmp_path / "one.mp4", 0, 2, False, "cut", 0)],
        tmp_path / "pad.mp4",
        1080,
        1920,
        fit_mode="pad",
    )
    pad_graph = pad[pad.index("-filter_complex") + 1]
    assert "force_original_aspect_ratio=decrease" in pad_graph
    assert "pad=width=1080:height=1920" in pad_graph


def test_shot_renderer_rejects_unknown_fit_mode(tmp_path):
    with pytest.raises(ValueError, match="fit_mode"):
        _build_shot_render_command(
            "ffmpeg",
            [(tmp_path / "one.mp4", 0, 2, False, "cut", 0)],
            tmp_path / "out.mp4",
            1920,
            1080,
            fit_mode="stretch",
        )


def test_shot_renderer_uses_concat_for_hard_cuts(tmp_path):
    command = _build_shot_render_command(
        "ffmpeg",
        [
            (tmp_path / "one.mp4", 0, 3, False, "cut", 0),
            (tmp_path / "two.mp4", 0, 2, False, "cut", 0),
        ],
        tmp_path / "out.mp4",
        1280,
        720,
    )
    graph = command[command.index("-filter_complex") + 1]
    assert "[v0][a0][v1][a1]concat=n=2:v=1:a=1" in graph
    assert "anullsrc=r=48000:cl=stereo" in graph


def test_render_requires_complete_human_approval(tmp_path):
    artifact = tmp_path / ".video-agent"
    artifact.mkdir()
    timeline = _timeline(tmp_path, approved=False)
    (artifact / "optimized-timeline.json").write_text(timeline.model_dump_json(), encoding="utf-8")

    with pytest.raises(RenderError, match="not been fully approved"):
        render_approved_shot_timeline(tmp_path)


def test_render_rejects_path_traversal_before_invoking_ffmpeg(tmp_path, monkeypatch):
    artifact = tmp_path / ".video-agent"
    artifact.mkdir()
    timeline = ProjectDraftTimeline(
        project_path=str(tmp_path.resolve()),
        style="cinematic",
        pacing="moderate",
        target_duration_seconds=3,
        proposed_duration_seconds=3,
        approved_duration_seconds=3,
        proposed_clip_count=1,
        clips=[
            DraftTimelineClip(
                clip_id="escape",
                sequence_index=1,
                video_path="../outside.mp4",
                source_start_seconds=0,
                source_end_seconds=3,
                duration_seconds=3,
                combined_score=90,
                pacing_fit="within",
                pacing_note="Within range.",
                decision="approved",
            )
        ],
        timeline_approved=True,
    )
    (artifact / "optimized-timeline.json").write_text(timeline.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        "video_edit_agent.rendering.shot_renderer.shutil.which",
        lambda _: "ffmpeg",
    )

    with pytest.raises(RenderError, match="escapes the project folder"):
        render_approved_shot_timeline(tmp_path)


def test_render_rejects_existing_output_without_overwrite(tmp_path):
    artifact = tmp_path / ".video-agent"
    artifact.mkdir()
    timeline = _timeline(tmp_path)
    (artifact / "optimized-timeline.json").write_text(timeline.model_dump_json(), encoding="utf-8")
    output = artifact / "rendered.mp4"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="pass --overwrite"):
        render_approved_shot_timeline(tmp_path, output_path=output)


def test_render_draft_cli_invokes_approved_renderer(tmp_path, monkeypatch):
    report = RenderReport(
        output_path=str(tmp_path / ".video-agent" / "rendered-draft.mp4"),
        source_timeline=str(tmp_path / ".video-agent" / "optimized-timeline.json"),
        source_video_count=1,
        segment_count=2,
        expected_duration_seconds=5.5,
        actual_duration_seconds=5.5,
        width=1920,
        height=1080,
        video_codec="h264",
        audio_codec="aac",
    )
    monkeypatch.setattr(
        "video_edit_agent.cli.rendering.render_approved_shot_timeline",
        lambda *args, **kwargs: report,
    )
    result = runner.invoke(app, ["render-draft", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "rendered and verified" in result.output


def test_cli_help_includes_render_draft():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "render-draft" in result.output
    result = runner.invoke(app, ["render-draft", "--help"])
    assert result.exit_code == 0
    assert "--fit-mode" in click.unstyle(result.output)
    assert "--music-mode" in click.unstyle(result.output)
    assert "--music-ducking" in click.unstyle(result.output)
    result = runner.invoke(app, ["render", "--help"])
    assert result.exit_code == 0
    assert "--music-mode" in click.unstyle(result.output)


def test_render_draft_cli_dry_run_reports_preflight_without_rendering(tmp_path, monkeypatch):
    calls = {}

    def fake_render(*args, **kwargs):
        calls.update(kwargs)
        return None

    monkeypatch.setattr("video_edit_agent.cli.rendering.render_approved_shot_timeline", fake_render)
    result = runner.invoke(app, ["render-draft", str(tmp_path), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert calls["dry_run"] is True
    assert "preflight passed" in result.output
    assert "no MP4 was written" in result.output


def test_approved_duration_does_not_apply_transition_across_rejected_shot(tmp_path):
    clips = [
        _clip("first", 1, 0, 3, decision="approved"),
        _clip(
            "middle",
            2,
            4,
            7,
            decision="rejected",
            transition="crossfade",
            transition_duration=0.5,
        ),
        _clip(
            "last",
            3,
            8,
            11,
            decision="approved",
            transition="crossfade",
            transition_duration=0.5,
        ),
    ]
    timeline = ProjectDraftTimeline(
        project_path=str(tmp_path.resolve()),
        style="cinematic",
        pacing="moderate",
        target_duration_seconds=10,
        proposed_duration_seconds=8,
        approved_duration_seconds=6,
        proposed_clip_count=3,
        clips=clips,
        timeline_approved=True,
    )
    assert timeline.approved_duration_seconds == 6


def test_render_approved_shot_timeline_with_real_ffmpeg(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required for the renderer integration test")

    first_source = tmp_path / "first.mp4"
    second_source = tmp_path / "second.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=30:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(first_source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=30:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(second_source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    artifact = tmp_path / ".video-agent"
    artifact.mkdir()
    timeline = ProjectDraftTimeline(
        project_path=str(tmp_path.resolve()),
        style="cinematic",
        pacing="moderate",
        target_duration_seconds=5,
        proposed_duration_seconds=3.6,
        approved_duration_seconds=3.6,
        proposed_clip_count=2,
        clips=[
            DraftTimelineClip(
                clip_id="first",
                sequence_index=1,
                video_path="first.mp4",
                source_start_seconds=0,
                source_end_seconds=2,
                duration_seconds=2,
                combined_score=90,
                pacing_fit="within",
                pacing_note="Within pacing range.",
                decision="approved",
            ),
            DraftTimelineClip(
                clip_id="second",
                sequence_index=2,
                video_path="second.mp4",
                source_start_seconds=0,
                source_end_seconds=2,
                duration_seconds=2,
                combined_score=85,
                transition_in="crossfade",
                transition_duration_seconds=0.4,
                pacing_fit="within",
                pacing_note="Within pacing range.",
                decision="approved",
            ),
        ],
        timeline_approved=True,
    )
    (artifact / "optimized-timeline.json").write_text(timeline.model_dump_json(), encoding="utf-8")

    preflight = render_approved_shot_timeline(tmp_path, dry_run=True)
    output = artifact / "rendered-draft.mp4"
    assert preflight is None
    assert not output.exists()

    report = render_approved_shot_timeline(tmp_path)
    assert output.is_file()
    assert report.video_codec == "h264"
    assert report.audio_codec == "aac"
    assert report.segment_count == 2
    assert report.source_video_count == 2
    assert report.actual_duration_seconds == pytest.approx(3.6, abs=0.2)
