from pathlib import Path

import pytest
from typer.testing import CliRunner

from travel_video_agent.cli import app
from travel_video_agent.editing.timeline_schemas import (
    DraftTimelineClip,
    ProjectDraftTimeline,
)
from travel_video_agent.rendering import RenderError, RenderReport, render_approved_shot_timeline
from travel_video_agent.rendering.shot_renderer import _build_shot_render_command

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
        approved_duration_seconds=5.5 if approved else 0,
        proposed_clip_count=2,
        clips=clips if approved else [
            clips[0],
            _clip("two", 2, 5, 8, decision="pending", transition="crossfade", transition_duration=0.5),
        ],
        timeline_approved=approved,
    )


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
    (artifact / "optimized-timeline.json").write_text(
        timeline.model_dump_json(), encoding="utf-8"
    )

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
    (artifact / "optimized-timeline.json").write_text(
        timeline.model_dump_json(), encoding="utf-8"
    )
    monkeypatch.setattr("travel_video_agent.rendering.shot_renderer.shutil.which", lambda _: "ffmpeg")

    with pytest.raises(RenderError, match="escapes the project folder"):
        render_approved_shot_timeline(tmp_path)


def test_render_rejects_existing_output_without_overwrite(tmp_path):
    artifact = tmp_path / ".video-agent"
    artifact.mkdir()
    timeline = _timeline(tmp_path)
    (artifact / "optimized-timeline.json").write_text(
        timeline.model_dump_json(), encoding="utf-8"
    )
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
        "travel_video_agent.cli.render_approved_shot_timeline",
        lambda *args, **kwargs: report,
    )
    result = runner.invoke(app, ["render-draft", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "rendered and verified" in result.output


def test_cli_help_includes_render_draft():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "render-draft" in result.output
