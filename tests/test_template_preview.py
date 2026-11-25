from pathlib import Path
from types import SimpleNamespace

import pytest

from video_edit_agent.editing.planning_schemas import StoryTreatment
from video_edit_agent.rendering.template_preview import (
    TemplatePreviewError,
    _opening_overlays,
    _probe_dimensions,
    render_template_previews,
)
from video_edit_agent.templates import load_template


def test_opening_overlays_use_template_slots_and_separate_logline():
    template, _ = load_template("cinematic")
    treatment = StoryTreatment(
        working_title="A Journey North",
        logline="A quiet trip through the mountains.",
        narrative_arc=["Departure", "Return"],
        emotional_journey="Wonder",
    )

    overlays = _opening_overlays(template, treatment)

    assert [overlay.text for overlay in overlays] == [
        "A Journey North",
        "A quiet trip through the mountains.",
    ]
    assert overlays[0].decision == "approved"
    assert overlays[0].placement != overlays[1].placement
    assert all(overlay.animation == "none" for overlay in overlays)


def test_opening_overlays_reject_template_without_opening_slots():
    template = SimpleNamespace(id="empty", sequences={})
    treatment = SimpleNamespace(working_title="Title", logline="Logline")

    with pytest.raises(ValueError, match="no usable opening title slots"):
        _opening_overlays(template, treatment)


def test_probe_dimensions_reads_ffprobe_output(monkeypatch, tmp_path):
    import video_edit_agent.rendering.template_preview as preview

    monkeypatch.setattr(
        preview,
        "run_ffmpeg",
        lambda *args, **kwargs: SimpleNamespace(stdout='{"streams":[{"width":1280,"height":720}]}'),
    )

    assert _probe_dimensions("ffprobe", tmp_path / "sample.jpg") == (1280, 720)


def test_probe_dimensions_rejects_missing_video_stream(monkeypatch, tmp_path):
    import video_edit_agent.rendering.template_preview as preview

    monkeypatch.setattr(
        preview,
        "run_ffmpeg",
        lambda *args, **kwargs: SimpleNamespace(stdout='{"streams":[]}'),
    )

    with pytest.raises(TemplatePreviewError, match="could not determine"):
        _probe_dimensions("ffprobe", tmp_path / "sample.jpg")


def test_template_preview_requires_unique_two_or_three_ids(tmp_path):
    with pytest.raises(ValueError, match="exactly two or three"):
        render_template_previews(tmp_path, ["cinematic"])

    with pytest.raises(ValueError, match="unique"):
        render_template_previews(tmp_path, ["cinematic", "cinematic"])


def test_template_preview_output_must_stay_inside_project(tmp_path):
    outside = tmp_path.parent / "outside-previews"

    with pytest.raises(ValueError, match="inside the project folder"):
        render_template_previews(tmp_path, ["cinematic", "vlog"], outside)


def _write_preview_project(
    root,
    *,
    timeline_approved=True,
    include_story=True,
    video_path="footage.mp4",
):
    from video_edit_agent.editing.intent_schemas import EditingPlan
    from video_edit_agent.editing.planning_schemas import (
        CreativeBrief,
        StoryTreatment,
        VideoProjectPlan,
    )
    from video_edit_agent.editing.timeline_schemas import (
        DraftTimelineClip,
        ProjectDraftTimeline,
    )

    metadata = root / ".video-agent"
    metadata.mkdir(parents=True, exist_ok=True)
    (root / "footage.mp4").write_bytes(b"source-video-placeholder")
    story = (
        StoryTreatment(
            working_title="A Journey North",
            logline="A quiet journey through the Canadian Rockies.",
            narrative_arc=["Departure", "Discovery"],
            emotional_journey="Curiosity to wonder",
        )
        if include_story
        else None
    )
    plan = VideoProjectPlan(
        project_path=str(root.resolve()),
        brief=CreativeBrief(
            source_prompt="Create a short cinematic travel film",
            target_duration_seconds=10,
            style="cinematic",
            aspect_ratio="16:9",
            resolution="1080p",
            pacing="moderate",
        ),
        editing_preferences=EditingPlan(source_prompt="Create a short cinematic travel film"),
        story_treatment=story,
        planning_stage="story_ready" if include_story else "brief_ready",
    )
    plan_path = metadata / "video-project-plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")

    clip = DraftTimelineClip(
        clip_id="clip-1",
        sequence_index=1,
        video_path=video_path,
        source_start_seconds=0,
        source_end_seconds=2,
        duration_seconds=2,
        combined_score=80,
        pacing_note="A brief establishing shot",
        decision="approved" if timeline_approved else "pending",
    )
    timeline = ProjectDraftTimeline(
        project_path=str(root.resolve()),
        style="cinematic",
        pacing="moderate",
        target_duration_seconds=10,
        proposed_duration_seconds=2,
        approved_duration_seconds=2 if timeline_approved else 0,
        proposed_clip_count=1,
        clips=[clip],
        timeline_approved=timeline_approved,
    )
    (metadata / "optimized-timeline.json").write_text(timeline.model_dump_json(), encoding="utf-8")
    return metadata


def test_template_preview_renders_comparison_and_writes_safe_index(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import video_edit_agent.rendering.template_preview as preview

    _write_preview_project(tmp_path)
    calls = []

    def fake_run_ffmpeg(command, **kwargs):
        calls.append(command)
        if Path(command[0]).name == "ffprobe":
            return SimpleNamespace(stdout='{"streams":[{"width":320,"height":180}]}')
        output = Path(command[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"rendered-preview")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(preview.shutil, "which", lambda executable: f"/usr/bin/{executable}")
    monkeypatch.setattr(preview, "run_ffmpeg", fake_run_ffmpeg)

    index, images, timestamp = render_template_previews(tmp_path, ["cinematic", "vlog"])

    assert index.is_file()
    assert [path.name for path in images] == ["cinematic.jpg", "vlog.jpg"]
    assert all(path.is_file() for path in images)
    assert timestamp == 1
    assert "Previews do not approve or apply templates" in index.read_text(encoding="utf-8")
    assert len([command for command in calls if command[0] == "/usr/bin/ffmpeg"]) == 3
    assert any("-filter_complex" in command for command in calls)


def test_template_preview_requires_existing_project_and_artifacts(tmp_path):
    missing_project = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="Project folder"):
        render_template_previews(missing_project, ["cinematic", "vlog"])

    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(FileNotFoundError, match="optimized timeline"):
        render_template_previews(project, ["cinematic", "vlog"])


def test_template_preview_requires_approved_timeline_and_story_treatment(tmp_path):
    unapproved = tmp_path / "unapproved"
    _write_preview_project(unapproved, timeline_approved=False)
    with pytest.raises(ValueError, match="must be approved"):
        render_template_previews(unapproved, ["cinematic", "vlog"])

    no_story = tmp_path / "no-story"
    _write_preview_project(no_story, include_story=False)
    with pytest.raises(ValueError, match="story treatment"):
        render_template_previews(no_story, ["cinematic", "vlog"])


@pytest.mark.parametrize(
    ("video_path", "message"),
    [
        ("/tmp/outside.mp4", "project-relative"),
        ("../outside.mp4", "escapes the project folder"),
        ("missing.mp4", "does not exist"),
    ],
)
def test_template_preview_rejects_unsafe_or_missing_source_paths(tmp_path, video_path, message):
    project = tmp_path / "project"
    _write_preview_project(project, video_path=video_path)

    with pytest.raises((ValueError, FileNotFoundError), match=message):
        render_template_previews(project, ["cinematic", "vlog"])


def test_template_preview_requires_ffmpeg_tools(monkeypatch, tmp_path):
    import video_edit_agent.rendering.template_preview as preview

    _write_preview_project(tmp_path)
    monkeypatch.setattr(preview.shutil, "which", lambda _executable: None)

    with pytest.raises(TemplatePreviewError, match="ffmpeg and ffprobe"):
        render_template_previews(tmp_path, ["cinematic", "vlog"])


def test_template_preview_rejects_existing_outputs_before_rendering(monkeypatch, tmp_path):
    import video_edit_agent.rendering.template_preview as preview

    _write_preview_project(tmp_path)
    destination = tmp_path / ".video-agent" / "template-previews"
    destination.mkdir()
    (destination / "cinematic.jpg").write_bytes(b"existing")
    monkeypatch.setattr(preview.shutil, "which", lambda executable: f"/usr/bin/{executable}")

    with pytest.raises(FileExistsError, match="already exists"):
        render_template_previews(tmp_path, ["cinematic", "vlog"])


def test_template_preview_fails_when_frame_extraction_produces_no_file(monkeypatch, tmp_path):
    import video_edit_agent.rendering.template_preview as preview

    _write_preview_project(tmp_path)
    monkeypatch.setattr(preview.shutil, "which", lambda executable: f"/usr/bin/{executable}")
    monkeypatch.setattr(preview, "run_ffmpeg", lambda *args, **kwargs: None)

    with pytest.raises(TemplatePreviewError, match="did not produce a preview frame"):
        render_template_previews(tmp_path, ["cinematic", "vlog"])


def test_template_preview_wraps_ffmpeg_render_errors(monkeypatch, tmp_path):
    import video_edit_agent.rendering.template_preview as preview

    _write_preview_project(tmp_path)
    calls = 0

    def fake_run_ffmpeg(command, **kwargs):
        nonlocal calls
        calls += 1
        if Path(command[0]).name == "ffprobe":
            from types import SimpleNamespace

            return SimpleNamespace(stdout='{"streams":[{"width":320,"height":180}]}')
        if calls == 1:
            Path(command[-1]).write_bytes(b"sample-frame")
            return None
        raise TemplatePreviewError("simulated filter render failure")

    monkeypatch.setattr(preview.shutil, "which", lambda executable: f"/usr/bin/{executable}")
    monkeypatch.setattr(preview, "run_ffmpeg", fake_run_ffmpeg)

    with pytest.raises(TemplatePreviewError, match="simulated filter render failure"):
        render_template_previews(tmp_path, ["cinematic", "vlog"])


def test_template_preview_renders_real_ffmpeg_frames_with_visible_overlays(tmp_path):
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("FFmpeg and FFprobe are required for this integration smoke test")

    _write_preview_project(tmp_path)
    source = tmp_path / "footage.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=12:duration=2",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(source),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )

    index, images, timestamp = render_template_previews(tmp_path, ["cinematic", "vlog"])
    baseline = tmp_path / "baseline.jpg"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-vf",
            "scale=1280:-2:force_original_aspect_ratio=decrease",
            "-q:v",
            "2",
            "-y",
            str(baseline),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )

    def decoded_pixels(path: Path) -> bytes:
        result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return result.stdout

    assert index.is_file()
    assert timestamp == 1
    assert baseline.is_file()
    baseline_pixels = decoded_pixels(baseline)
    assert len(baseline_pixels) == 1280 * 720 * 3
    for image_path in images:
        assert image_path.is_file()
        assert image_path.stat().st_size > 0
        assert _probe_dimensions(ffprobe, image_path) == (1280, 720)
        assert decoded_pixels(image_path) != baseline_pixels
