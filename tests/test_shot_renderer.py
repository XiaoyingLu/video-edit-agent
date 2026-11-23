import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
import typer.rich_utils
from typer.testing import CliRunner

from video_edit_agent.artifact_tracking import register_artifact
from video_edit_agent.cli import app
from video_edit_agent.editing.timeline_schemas import (
    DraftTimelineClip,
    ProjectDraftTimeline,
)
from video_edit_agent.rendering import RenderError, RenderReport, render_approved_shot_timeline
from video_edit_agent.rendering.shot_renderer import (
    _build_beat_zoom_paths,
    _build_shot_render_command,
    _requested_output_dimensions,
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


@pytest.mark.parametrize(
    ("transition", "xfade_name"),
    [
        ("fadeblack", "fadeblack"),
        ("wipeleft", "wipeleft"),
        ("slideleft", "slideleft"),
        ("circleopen", "circleopen"),
    ],
)
def test_shot_renderer_builds_additional_xfade_transitions(tmp_path, transition, xfade_name):
    command = _build_shot_render_command(
        "ffmpeg",
        [
            (tmp_path / "one.mp4", 0, 3, True, "cut", 0),
            (tmp_path / "two.mp4", 5, 8, True, transition, 0.5),
        ],
        tmp_path / "out.mp4",
        1920,
        1080,
    )
    graph = command[command.index("-filter_complex") + 1]
    assert f"xfade=transition={xfade_name}:duration=0.500000" in graph


def test_face_reframing_uses_detected_face_position(tmp_path, monkeypatch):
    import numpy as np

    class FakeCapture:
        def isOpened(self):
            return True

        def set(self, *_args):
            return True

        def read(self):
            return True, np.zeros((100, 200, 3), dtype=np.uint8)

        def release(self):
            return None

    class FakeDetector:
        def empty(self):
            return False

        def detectMultiScale(self, *_args, **_kwargs):
            return [(40, 0, 40, 40)]

    fake_cv2 = SimpleNamespace(
        data=SimpleNamespace(haarcascades="/fake/"),
        CascadeClassifier=lambda _path: FakeDetector(),
        VideoCapture=lambda _path: FakeCapture(),
        CAP_PROP_POS_MSEC=0,
        COLOR_BGR2GRAY=1,
        resize=lambda frame, _size: np.zeros((50, 100, 3), dtype=np.uint8),
        cvtColor=lambda frame, _code: frame[:, :, 0],
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    from video_edit_agent.rendering.face_reframing import detect_face_crop_anchors

    anchors = detect_face_crop_anchors([(tmp_path / "clip.mp4", 0, 4, False, "cut", 0)])
    assert len(anchors) == 1
    assert anchors[0] == pytest.approx((0.6, 0.4))


def test_shot_renderer_uses_normalized_face_crop_anchor(tmp_path):
    command = _build_shot_render_command(
        "ffmpeg",
        [(tmp_path / "one.mp4", 0, 2, False, "cut", 0)],
        tmp_path / "out.mp4",
        1080,
        1920,
        crop_anchors=[(0.72, 0.38)],
    )
    graph = command[command.index("-filter_complex") + 1]
    assert "x='max(0,min(iw-ow,iw*(0.720000)-ow/2))'" in graph
    assert "y='max(0,min(ih-oh,ih*(0.380000)-oh/2))'" in graph


def test_shot_renderer_rejects_invalid_crop_anchors(tmp_path):
    with pytest.raises(ValueError, match="normalized"):
        _build_shot_render_command(
            "ffmpeg",
            [(tmp_path / "one.mp4", 0, 2, False, "cut", 0)],
            tmp_path / "out.mp4",
            1080,
            1920,
            crop_anchors=[(1.2, 0.5)],
        )


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
        "video_edit_agent.cli.rendering_core.render_approved_shot_timeline",
        lambda *args, **kwargs: report,
    )
    result = runner.invoke(app, ["render-draft", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "rendered and verified" in result.output


def test_cli_help_includes_render_draft():
    result = runner.invoke(app, ["render", "--help"])
    assert result.exit_code == 0
    assert "render-draft" in result.output
    assert "style-presets" in result.output
    result = runner.invoke(app, ["render-draft", "--help"])
    assert result.exit_code == 0
    assert "--style-preset" in click.unstyle(result.output)
    assert "--fit-mode" in click.unstyle(result.output)
    assert "--music-mode" in click.unstyle(result.output)
    assert "--music-ducking" in click.unstyle(result.output)
    assert "--aspect-ratio" in click.unstyle(result.output)
    assert "--resolution" in click.unstyle(result.output)
    result = runner.invoke(app, ["render", "video", "--help"])
    assert result.exit_code == 0
    assert "--music-mode" in click.unstyle(result.output)


def test_render_draft_cli_dry_run_reports_preflight_without_rendering(tmp_path, monkeypatch):
    calls = {}

    def fake_render(*args, **kwargs):
        calls.update(kwargs)
        return None

    monkeypatch.setattr(
        "video_edit_agent.cli.rendering_core.render_approved_shot_timeline", fake_render
    )
    result = runner.invoke(app, ["render-draft", str(tmp_path), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert calls["dry_run"] is True
    assert "preflight passed" in result.output
    assert "no MP4 was written" in result.output


def test_render_draft_cli_applies_preset_and_explicit_overrides(tmp_path, monkeypatch):
    calls = {}

    def fake_render(*args, **kwargs):
        calls.update(kwargs)
        return None

    monkeypatch.setattr(
        "video_edit_agent.cli.rendering_core.render_approved_shot_timeline", fake_render
    )
    result = runner.invoke(
        app,
        [
            "render-draft",
            str(tmp_path),
            "--dry-run",
            "--style-preset",
            "cinematic",
            "--music-volume",
            "0.3",
            "--no-music-ducking",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["fit_mode"] == "crop"
    assert calls["music_volume"] == 0.3
    assert calls["music_fade_in"] == 1.8
    assert calls["music_fade_out"] == 2.5
    assert calls["music_ducking"] is False


def test_render_draft_cli_passes_social_output_profile(tmp_path, monkeypatch):
    calls = {}

    def fake_render(*args, **kwargs):
        calls.update(kwargs)
        return None

    monkeypatch.setattr(
        "video_edit_agent.cli.rendering_core.render_approved_shot_timeline", fake_render
    )
    result = runner.invoke(
        app,
        [
            "render-draft",
            str(tmp_path),
            "--dry-run",
            "--aspect-ratio",
            "9:16",
            "--resolution",
            "1080p",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["output_aspect_ratio"] == "9:16"
    assert calls["output_resolution"] == "1080p"


def test_render_draft_requires_aspect_ratio_and_resolution_together(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "video_edit_agent.cli.rendering_core.render_approved_shot_timeline",
        lambda *args, **kwargs: None,
    )
    result = runner.invoke(
        app,
        ["render-draft", str(tmp_path), "--aspect-ratio", "9:16", "--dry-run"],
    )
    assert result.exit_code == 1
    assert "--aspect-ratio and --resolution must be supplied together" in result.output


def test_render_social_exports_generates_three_preflights(tmp_path, monkeypatch):
    calls = []

    def fake_render(*args, **kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(
        "video_edit_agent.cli.rendering_core.render_approved_shot_timeline", fake_render
    )
    overlay_file = tmp_path / ".video-agent" / "title-overlays.json"
    result = runner.invoke(
        app,
        [
            "render-social-exports",
            str(tmp_path),
            "--dry-run",
            "--title-overlays",
            str(overlay_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 3
    assert [call["output_aspect_ratio"] for call in calls] == ["16:9", "9:16", "1:1"]
    assert {call["output_resolution"] for call in calls} == {"1080p"}
    assert all(call["dry_run"] for call in calls)
    assert all(call["title_overlay_path"] == overlay_file for call in calls)
    assert "No MP4 files were written" in result.output


def test_social_exports_cli_passes_face_reframing_option(tmp_path, monkeypatch):
    calls = []

    def fake_render(*args, **kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(
        "video_edit_agent.cli.rendering_core.render_approved_shot_timeline", fake_render
    )
    result = runner.invoke(
        app,
        ["render-social-exports", str(tmp_path), "--dry-run", "--reframe", "face"],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 3
    assert all(call["reframe_mode"] == "face" for call in calls)


def test_social_exports_cli_rejects_face_reframing_with_pad(tmp_path):
    result = runner.invoke(
        app,
        [
            "render-social-exports",
            str(tmp_path),
            "--dry-run",
            "--fit-mode",
            "pad",
            "--reframe",
            "face",
        ],
    )

    assert result.exit_code == 1
    assert "--reframe face requires --fit-mode crop" in result.output


def test_social_exports_cli_rejects_crop_anchors_with_pad(tmp_path):
    result = runner.invoke(
        app,
        [
            "render-social-exports",
            str(tmp_path),
            "--dry-run",
            "--fit-mode",
            "pad",
            "--crop-anchors",
            str(tmp_path / "anchors.json"),
        ],
    )

    assert result.exit_code == 1
    assert "--crop-anchors requires --fit-mode crop" in result.output


def test_style_presets_cli_lists_available_profiles():
    result = runner.invoke(app, ["style-presets"])
    assert result.exit_code == 0, result.output
    assert "cinematic" in result.output
    assert "vlog" in result.output
    assert "travel-montage" in result.output
    assert "social" in result.output
    assert "Beat zoom" in result.output


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


@pytest.mark.parametrize(
    ("aspect_ratio", "resolution", "expected"),
    [
        ("16:9", "720p", (1280, 720)),
        ("16:9", "1080p", (1920, 1080)),
        ("9:16", "1080p", (1080, 1920)),
        ("1:1", "4k", (2160, 2160)),
    ],
)
def test_requested_output_dimensions(aspect_ratio, resolution, expected):
    assert _requested_output_dimensions(aspect_ratio, resolution) == expected


@pytest.mark.parametrize(
    ("aspect_ratio", "resolution"),
    [
        ("4:3", "1080p"),
        ("16:9", "480p"),
    ],
)
def test_requested_output_dimensions_reject_unsupported_profiles(aspect_ratio, resolution):
    with pytest.raises(ValueError):
        _requested_output_dimensions(aspect_ratio, resolution)


@pytest.mark.parametrize("color", [False, True])
def test_create_video_help_exposes_single_final_review_option(tmp_path, monkeypatch, color):
    monkeypatch.setattr(typer.rich_utils, "FORCE_TERMINAL", color)
    result = runner.invoke(
        app,
        ["create-video", str(tmp_path), "--help"],
        color=True,
        env={"TERM": "xterm", "NO_COLOR": None},
    )
    assert result.exit_code == 0
    assert ("\x1b[" in result.output) is color
    help_text = click.unstyle(result.output)
    assert "--review" in help_text
    assert "--storyboard" in help_text
    assert "automatically assembled" in " ".join(help_text.replace("│", " ").split())


def test_manual_crop_anchor_template_is_timeline_bound_and_overrides_face(tmp_path):
    from video_edit_agent.rendering.social_crop_anchors import (
        create_crop_anchor_template,
        load_crop_anchors,
    )

    artifact = tmp_path / ".video-agent"
    artifact.mkdir()
    timeline_file = artifact / "optimized-timeline.json"
    timeline = _timeline(tmp_path)
    timeline_file.write_text(timeline.model_dump_json(), encoding="utf-8")
    template, clip_ids = create_crop_anchor_template(tmp_path)
    document = json.loads(template.read_text(encoding="utf-8"))
    assert clip_ids == ["one", "two"]
    document["anchors"] = [{"clip_id": "two", "x": 0.72, "y": 0.38}]
    template.write_text(json.dumps(document), encoding="utf-8")

    # Relative paths are interpreted from the project root, not the process cwd.
    anchors = load_crop_anchors(
        tmp_path,
        timeline_file,
        timeline,
        Path(".video-agent/social-crop-anchors.json"),
    )
    assert anchors == {"two": (0.72, 0.38)}


def test_manual_crop_anchor_rejects_stale_timeline(tmp_path):
    from video_edit_agent.rendering.social_crop_anchors import (
        create_crop_anchor_template,
        load_crop_anchors,
    )

    artifact = tmp_path / ".video-agent"
    artifact.mkdir()
    timeline_file = artifact / "optimized-timeline.json"
    timeline = _timeline(tmp_path)
    timeline_file.write_text(timeline.model_dump_json(), encoding="utf-8")
    template, _ = create_crop_anchor_template(tmp_path)
    timeline_file.write_text(timeline.model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        load_crop_anchors(tmp_path, timeline_file, timeline, template)


def test_manual_crop_anchor_rejects_unknown_duplicate_and_invalid_values(tmp_path):
    from video_edit_agent.rendering.social_crop_anchors import (
        create_crop_anchor_template,
        load_crop_anchors,
    )

    artifact = tmp_path / ".video-agent"
    artifact.mkdir()
    timeline_file = artifact / "optimized-timeline.json"
    timeline = _timeline(tmp_path)
    timeline_file.write_text(timeline.model_dump_json(), encoding="utf-8")
    template, _ = create_crop_anchor_template(tmp_path)
    base = json.loads(template.read_text(encoding="utf-8"))
    cases = [
        ([{"clip_id": "missing", "x": 0.5, "y": 0.5}], "unknown"),
        ([{"clip_id": "one", "x": 0.5, "y": 0.5}] * 2, "duplicate"),
        ([{"clip_id": "one", "x": 1.2, "y": 0.5}], "between 0 and 1"),
    ]
    for entries, message in cases:
        document = {**base, "anchors": entries}
        template.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_crop_anchors(tmp_path, timeline_file, timeline, template)


def test_init_social_crop_anchors_cli_creates_template(tmp_path):
    artifact = tmp_path / ".video-agent"
    artifact.mkdir()
    (artifact / "optimized-timeline.json").write_text(
        _timeline(tmp_path).model_dump_json(), encoding="utf-8"
    )
    result = runner.invoke(app, ["init-social-crop-anchors", str(tmp_path)])
    assert result.exit_code == 0, result.output
    template = json.loads((artifact / "social-crop-anchors.json").read_text(encoding="utf-8"))
    assert template["available_clip_ids"] == ["one", "two"]
    assert template["anchors"] == []


def test_social_exports_cli_passes_manual_crop_anchors(tmp_path, monkeypatch):
    calls = []

    def fake_render(*args, **kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(
        "video_edit_agent.cli.rendering_core.render_approved_shot_timeline", fake_render
    )
    anchor_file = tmp_path / "anchors.json"
    result = runner.invoke(
        app,
        [
            "render-social-exports",
            str(tmp_path),
            "--dry-run",
            "--crop-anchors",
            str(anchor_file),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 3
    assert all(call["crop_anchors_path"] == anchor_file for call in calls)


def test_shot_renderer_builds_time_varying_face_crop_path(tmp_path):
    command = _build_shot_render_command(
        "ffmpeg",
        [(tmp_path / "one.mp4", 0, 2, False, "cut", 0)],
        tmp_path / "out.mp4",
        1080,
        1920,
        crop_paths=[[(0.0, 0.25, 0.4), (1.0, 0.75, 0.6), (2.0, 0.6, 0.5)]],
    )
    graph = command[command.index("-filter_complex") + 1]
    assert "if(lt(t,1.000000)" in graph
    assert "iw*(if(" in graph
    assert "ih*(if(" in graph


def test_shot_renderer_builds_centered_beat_zoom_filter(tmp_path):
    command = _build_shot_render_command(
        "ffmpeg",
        [(tmp_path / "one.mp4", 0, 2, False, "cut", 0)],
        tmp_path / "out.mp4",
        1080,
        1920,
        zoom_paths=[[(0.0, 1.0, 1.0), (0.5, 1.06, 1.06), (0.74, 1.0, 1.0), (2.0, 1.0, 1.0)]],
    )
    graph = command[command.index("-filter_complex") + 1]
    assert "eval=frame" in graph
    assert "1.060000" in graph
    assert "crop=1080:1920:(iw-ow)/2:(ih-oh)/2" in graph


def test_beat_zoom_paths_follow_music_beats_and_transition_overlaps(tmp_path):
    from video_edit_agent.artifact_tracking import register_artifact
    from video_edit_agent.music.schemas import MusicAnalysis

    artifact = tmp_path / ".video-agent"
    artifact.mkdir()
    soundtrack = tmp_path / "music.wav"
    soundtrack.write_bytes(b"audio")
    analysis_path = artifact / "music-analysis.json"
    analysis = MusicAnalysis(
        project_path=str(tmp_path.resolve()),
        music_path=str(soundtrack.resolve()),
        duration_seconds=10,
        sample_rate=44100,
        bpm=120,
        beat_confidence=0.9,
        beat_interval_seconds=0.5,
        beat_times_seconds=[0.5, 1.0, 1.5, 2.0],
    )
    analysis_path.write_text(analysis.model_dump_json(), encoding="utf-8")
    register_artifact(tmp_path, analysis_path, dependencies=[soundtrack])
    clips = [
        (tmp_path / "a.mp4", 0, 1.5, False, "cut", 0),
        (tmp_path / "b.mp4", 0, 1.5, False, "crossfade", 0.5),
    ]

    paths = _build_beat_zoom_paths(tmp_path, soundtrack, clips)

    assert paths is not None
    assert len(paths) == 2
    assert any(point[1] == 1.06 for point in paths[0])
    assert any(point[1] == 1.06 for point in paths[1])


def test_beat_zoom_paths_skip_missing_music_analysis(tmp_path):
    soundtrack = tmp_path / "music.wav"
    soundtrack.write_bytes(b"audio")
    clips = [(tmp_path / "a.mp4", 0, 2, False, "cut", 0)]
    assert _build_beat_zoom_paths(tmp_path, soundtrack, clips) is None


def test_shot_renderer_rejects_crop_paths_for_wrong_clip_count(tmp_path):
    with pytest.raises(ValueError, match="one path per clip"):
        _build_shot_render_command(
            "ffmpeg",
            [
                (tmp_path / "one.mp4", 0, 2, False, "cut", 0),
                (tmp_path / "two.mp4", 0, 2, False, "cut", 0),
            ],
            tmp_path / "out.mp4",
            1080,
            1920,
            crop_paths=[[(0.0, 0.5, 0.5)]],
        )


def test_face_reframing_returns_smoothed_per_clip_paths(tmp_path, monkeypatch):
    import numpy as np

    class FakeCapture:
        def __init__(self):
            self.read_count = 0

        def isOpened(self):
            return True

        def set(self, *_args):
            return True

        def read(self):
            self.read_count += 1
            return True, np.zeros((100, 200, 3), dtype=np.uint8)

        def release(self):
            return None

    class FakeDetector:
        def empty(self):
            return False

        def detectMultiScale(self, *_args, **_kwargs):
            return [(40, 0, 40, 40)]

    fake_cv2 = SimpleNamespace(
        data=SimpleNamespace(haarcascades="/fake/"),
        CascadeClassifier=lambda _path: FakeDetector(),
        VideoCapture=lambda _path: FakeCapture(),
        CAP_PROP_POS_MSEC=0,
        COLOR_BGR2GRAY=1,
        resize=lambda frame, _size: np.zeros((50, 100, 3), dtype=np.uint8),
        cvtColor=lambda frame, _code: frame[:, :, 0],
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    from video_edit_agent.rendering.face_reframing import detect_face_crop_paths

    paths = detect_face_crop_paths([(tmp_path / "clip.mp4", 0, 2, False, "cut", 0)])
    assert len(paths) == 1
    assert len(paths[0]) >= 4
    assert paths[0][0][0] == 0.0
    assert paths[0][0][1:] == pytest.approx((0.6, 0.4))
