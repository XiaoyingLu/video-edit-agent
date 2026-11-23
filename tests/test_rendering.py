import json
import subprocess
from pathlib import Path

import pytest

from video_edit_agent.editing.schemas import (
    ProjectTimeline,
    TimelineSegment,
    VideoTimeline,
)
from video_edit_agent.rendering import RenderError, render_project


def _write_timeline(
    root: Path,
    *,
    source_path: str = "clip.mp4",
    duration: float = 10,
    segments: list[tuple[float, float]] | None = None,
) -> Path:
    (root / "clip.mp4").write_bytes(b"original-source")
    selected = segments if segments is not None else [(0, 2), (4, 10)]
    timeline = ProjectTimeline(
        project_path=str(root.resolve()),
        approved_cut_count=1,
        retained_duration_seconds=sum(end - start for start, end in selected),
        videos=[
            VideoTimeline(
                path=source_path,
                source_duration_seconds=duration,
                retained_duration_seconds=sum(end - start for start, end in selected),
                segments=[
                    TimelineSegment(start_seconds=start, end_seconds=end) for start, end in selected
                ],
            )
        ],
    )
    artifact_dir = root / ".video-agent"
    artifact_dir.mkdir(exist_ok=True)
    destination = artifact_dir / "timeline.json"
    destination.write_text(timeline.model_dump_json(), encoding="utf-8")
    return destination


def _fake_tools(monkeypatch, *, output_duration: float = 8, has_audio: bool = True):
    from video_edit_agent.rendering import renderer

    monkeypatch.setattr(
        renderer.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"ffmpeg", "ffprobe"} else None,
    )
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(command)
        if command[0].endswith("ffmpeg"):
            if command[-1] != "-":
                Path(command[-1]).write_bytes(b"rendered-mp4")
            return subprocess.CompletedProcess(command, 0, "", "")
        inspected = Path(command[-1])
        if inspected.name == "clip.mp4":
            streams = [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 640,
                    "height": 360,
                }
            ]
            if has_audio:
                streams.append({"codec_type": "audio", "codec_name": "aac"})
            payload = {"format": {"duration": "10.0"}, "streams": streams}
        else:
            payload = {
                "format": {"duration": str(output_duration)},
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 640,
                        "height": 360,
                    },
                    {"codec_type": "audio", "codec_name": "aac"},
                ],
            }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr("video_edit_agent.media_runner.subprocess.run", fake_run)
    return commands


def test_render_project_creates_verified_mp4_without_changing_source(
    tmp_path: Path, monkeypatch
) -> None:
    _write_timeline(tmp_path)
    commands = _fake_tools(monkeypatch)
    source = tmp_path / "clip.mp4"
    original = source.read_bytes()

    report = render_project(tmp_path)

    output = tmp_path / ".video-agent" / "rendered.mp4"
    assert output.read_bytes() == b"rendered-mp4"
    assert source.read_bytes() == original
    assert report.actual_duration_seconds == 8
    assert report.segment_count == 2
    assert report.video_codec == "h264"
    assert report.audio_codec == "aac"
    ffmpeg_command = next(command for command in commands if command[0].endswith("ffmpeg"))
    filter_graph = ffmpeg_command[ffmpeg_command.index("-filter_complex") + 1]
    assert "trim=start=0.000000:end=2.000000" in filter_graph
    assert "concat=n=2:v=1:a=1" in filter_graph
    assert "pad=width=640:height=360:x=(ow-iw)/2:y=(oh-ih)/2" in filter_graph
    assert "-c:v" in ffmpeg_command and "libx264" in ffmpeg_command


def test_render_project_dry_run_validates_graph_without_creating_output(
    tmp_path: Path, monkeypatch
) -> None:
    _write_timeline(tmp_path)
    commands = _fake_tools(monkeypatch)

    report = render_project(tmp_path, dry_run=True)

    output = tmp_path / ".video-agent" / "rendered.mp4"
    assert report is None
    assert not output.exists()
    validation_command = next(command for command in commands if command[0].endswith("ffmpeg"))
    assert validation_command[-5:] == ["-t", "0.5", "-f", "null", "-"]
    assert "-filter_complex" in validation_command
    assert "-c:v" not in validation_command


def test_render_project_burns_subtitles_after_concatenating_timeline(
    tmp_path: Path, monkeypatch
) -> None:
    _write_timeline(tmp_path)
    commands = _fake_tools(monkeypatch)
    subtitle_file = tmp_path / ".video-agent" / "subtitles.srt"
    subtitle_file.write_text("1\\n00:00:01,000 --> 00:00:02,000\\nHello\\n", encoding="utf-8")

    render_project(tmp_path, subtitle_path=subtitle_file, subtitle_font_size=32)

    ffmpeg_command = next(command for command in commands if command[0].endswith("ffmpeg"))
    filter_graph = ffmpeg_command[ffmpeg_command.index("-filter_complex") + 1]
    assert "concat=n=2:v=1:a=1[outv][outa];[outv]subtitles=" in filter_graph
    assert "FontSize=32" in filter_graph
    assert "-map" in ffmpeg_command
    assert ffmpeg_command[ffmpeg_command.index("-map") + 1] == "[subtitled]"


def test_render_project_generates_silence_for_source_without_audio(
    tmp_path: Path, monkeypatch
) -> None:
    _write_timeline(tmp_path, segments=[(0, 10)])
    commands = _fake_tools(monkeypatch, output_duration=10, has_audio=False)

    report = render_project(tmp_path)

    ffmpeg_command = next(command for command in commands if command[0].endswith("ffmpeg"))
    filter_graph = ffmpeg_command[ffmpeg_command.index("-filter_complex") + 1]
    assert "anullsrc=r=48000:cl=stereo" in filter_graph
    assert report.audio_codec == "aac"


def test_render_project_refuses_to_overwrite_existing_output(tmp_path: Path, monkeypatch) -> None:
    _write_timeline(tmp_path)
    commands = _fake_tools(monkeypatch)
    output = tmp_path / ".video-agent" / "rendered.mp4"
    output.write_bytes(b"keep-me")

    with pytest.raises(FileExistsError, match="--overwrite"):
        render_project(tmp_path)

    assert output.read_bytes() == b"keep-me"
    assert not any(command[0].endswith("ffmpeg") for command in commands)


def test_render_project_rejects_source_path_outside_project(tmp_path: Path, monkeypatch) -> None:
    outside = tmp_path.parent / "outside.mp4"
    outside.write_bytes(b"outside")
    _write_timeline(tmp_path, source_path="../outside.mp4")
    _fake_tools(monkeypatch)

    with pytest.raises(RenderError, match="escapes the project"):
        render_project(tmp_path)


def test_render_project_rejects_changed_source_duration(tmp_path: Path, monkeypatch) -> None:
    _write_timeline(tmp_path, duration=9, segments=[(0, 2), (4, 9)])
    _fake_tools(monkeypatch)

    with pytest.raises(RenderError, match="duration changed"):
        render_project(tmp_path)


def test_render_project_rejects_invalid_output_duration_and_cleans_temp_file(
    tmp_path: Path, monkeypatch
) -> None:
    _write_timeline(tmp_path)
    _fake_tools(monkeypatch, output_duration=5)

    with pytest.raises(RenderError, match="duration mismatch"):
        render_project(tmp_path)

    assert not (tmp_path / ".video-agent" / "rendered.mp4").exists()
    assert list((tmp_path / ".video-agent").glob(".video-agent-render-*.mp4")) == []


def test_render_project_refuses_output_path_equal_to_source(tmp_path: Path, monkeypatch) -> None:
    _write_timeline(tmp_path)
    _fake_tools(monkeypatch)

    with pytest.raises(RenderError, match="cannot overwrite a source"):
        render_project(tmp_path, output_path=tmp_path / "clip.mp4")


def test_render_project_requires_a_validated_timeline(tmp_path: Path, monkeypatch) -> None:
    _fake_tools(monkeypatch)
    (tmp_path / ".video-agent").mkdir()
    (tmp_path / ".video-agent" / "timeline.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        render_project(tmp_path)
