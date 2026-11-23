import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError
from rich.text import Text
from typer.testing import CliRunner

from video_edit_agent.cli import app
from video_edit_agent.editing.intent_schemas import EditingPlan
from video_edit_agent.editing.planning_schemas import (
    CreativeBrief,
    PlannedScene,
    StoryboardShot,
    StoryTreatment,
    VideoProjectPlan,
)
from video_edit_agent.editing.timeline_schemas import DraftTimelineClip, ProjectDraftTimeline
from video_edit_agent.rendering.critique import (
    AudioQualitySummary,
    CritiqueFinding,
    DraftCritiqueError,
    DraftCritiqueReport,
    FrameQualitySample,
    SemanticFrameReview,
    _parse_frame_rate,
    _quality_critique_findings,
    _semantic_critique_findings,
    critique_rendered_draft,
)

runner = CliRunner()


def _write_inputs(root: Path, *, target_duration: float = 10) -> tuple[Path, Path, Path]:
    artifact = root / ".video-agent"
    artifact.mkdir(parents=True)
    video = artifact / "rendered-draft.mp4"
    video.write_bytes(b"fake mp4")
    source = root / "source.mp4"
    source.write_bytes(b"source")

    brief = CreativeBrief(
        source_prompt="Create a short cinematic travel film",
        target_duration_seconds=target_duration,
        style="cinematic",
        aspect_ratio="16:9",
        resolution="1080p",
        pacing="moderate",
        creative_priorities=["scenic_landscapes"],
    )
    editing = EditingPlan(
        source_prompt=brief.source_prompt,
        target_duration_seconds=target_duration,
    )
    shot = StoryboardShot(
        shot_id="shot-1",
        scene_id="scene-1",
        sequence_index=1,
        purpose="Establish the destination",
        visual_description="A wide view of the mountains",
        target_duration_seconds=target_duration,
    )
    plan = VideoProjectPlan(
        project_path=str(root.resolve()),
        brief=brief,
        editing_preferences=editing,
        story_treatment=StoryTreatment(
            working_title="Mountain journey",
            logline="A short journey through mountain scenery.",
            narrative_arc=["Arrival", "Discovery"],
            emotional_journey="Curiosity to wonder",
        ),
        scenes=[
            PlannedScene(
                scene_id="scene-1",
                sequence_index=1,
                title="Arrival",
                purpose="Introduce the landscape",
                narrative_function="establishing",
                target_duration_seconds=target_duration,
                shots=[shot],
            )
        ],
        planning_stage="storyboard_ready",
    )
    plan_path = artifact / "video-project-plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")

    clip = DraftTimelineClip(
        clip_id="shot-1",
        sequence_index=1,
        video_path="source.mp4",
        source_start_seconds=0,
        source_end_seconds=target_duration,
        duration_seconds=target_duration,
        combined_score=90,
        pacing_fit="within",
        pacing_note="Within the recommended range.",
        decision="approved",
    )
    timeline = ProjectDraftTimeline(
        project_path=str(root.resolve()),
        style="cinematic",
        pacing="moderate",
        target_duration_seconds=target_duration,
        proposed_duration_seconds=target_duration,
        approved_duration_seconds=target_duration,
        proposed_clip_count=1,
        clips=[clip],
        timeline_approved=True,
    )
    timeline_path = artifact / "optimized-timeline.json"
    timeline_path.write_text(timeline.model_dump_json(), encoding="utf-8")
    return video, timeline_path, plan_path


def _frame(timestamp: float = 5) -> FrameQualitySample:
    return FrameQualitySample(
        timestamp_seconds=timestamp,
        mean_luma=110,
        dark_pixel_ratio=0.1,
        bright_pixel_ratio=0.05,
        sharpness_variance=125,
    )


def test_critique_compares_render_to_brief_and_approved_timeline(tmp_path, monkeypatch):
    video, timeline, plan = _write_inputs(tmp_path)
    monkeypatch.setattr("video_edit_agent.rendering.critique.validate_artifact", lambda *_: False)
    monkeypatch.setattr("video_edit_agent.rendering.critique.shutil.which", lambda name: name)

    probe = {
        "format": {"duration": "15"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30/1",
            },
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    }

    def fake_run(command, **kwargs):
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 0, json.dumps(probe), "")
        return subprocess.CompletedProcess(
            command,
            0,
            "",
            "mean_volume: -18.0 dB\nmax_volume: -2.0 dB\nsilence_start: 4.0\nsilence_end: 8.0",
        )

    monkeypatch.setattr("video_edit_agent.rendering.critique._run", fake_run)
    monkeypatch.setattr(
        "video_edit_agent.rendering.critique._sample_frames",
        lambda *_: [_frame(5), _frame(10)],
    )

    report = critique_rendered_draft(tmp_path, sample_count=2)

    assert report.rendered_video == video.relative_to(tmp_path).as_posix()
    assert report.source_timeline == timeline.relative_to(tmp_path).as_posix()
    assert report.source_plan == plan.relative_to(tmp_path).as_posix()
    assert report.duration_seconds == 15
    assert report.frame_rate == 30
    assert report.audio.mean_volume_db == -18
    assert report.audio.silence_intervals == [(4, 8)]
    assert report.timeline_modified is False
    assert report.rendered_video_modified is False
    assert {finding.category for finding in report.findings} >= {
        "duration",
        "render",
        "audio",
    }


def test_semantic_critique_compares_rendered_frames_to_storyboard(tmp_path, monkeypatch):
    _write_inputs(tmp_path)
    monkeypatch.setattr("video_edit_agent.rendering.critique.validate_artifact", lambda *_: False)
    monkeypatch.setattr("video_edit_agent.rendering.critique.shutil.which", lambda name: name)
    probe = {
        "format": {"duration": "10"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30/1",
            }
        ],
    }

    def fake_run(command, **kwargs):
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 0, json.dumps(probe), "")
        Path(command[-1]).write_bytes(b"frame")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("video_edit_agent.rendering.critique._run", fake_run)
    monkeypatch.setattr(
        "video_edit_agent.rendering.critique._sample_frames",
        lambda *_: [_frame(5)],
    )

    class FakeAnalyzer:
        def __init__(self, model):
            self.model = model
            self.comparison = None
            self.unloaded = False

        def compare_frame_to_storyboard(self, frame_path, **kwargs):
            assert frame_path.is_file()
            self.comparison = kwargs
            return {
                "match": "mismatch",
                "score": 0.2,
                "confidence": 0.9,
                "evidence": "The frame shows a close-up of food, not a mountain landscape.",
                "suggestion": "Review the footage assignment for this shot.",
            }

        def unload_model(self):
            self.unloaded = True
            return True

    analyzers = []

    def analyzer_factory(*args, **kwargs):
        analyzer = FakeAnalyzer(*args, **kwargs)
        analyzers.append(analyzer)
        return analyzer

    monkeypatch.setattr(
        "video_edit_agent.understanding.visual_analyzer.OllamaVisionAnalyzer",
        analyzer_factory,
    )

    report = critique_rendered_draft(
        tmp_path,
        semantic=True,
        semantic_samples=1,
        vision_model="qwen2.5vl:3b",
    )

    assert report.semantic_model == "qwen2.5vl:3b"
    assert len(report.semantic_reviews) == 1
    assert report.semantic_reviews[0].shot_id == "shot-1"
    assert report.semantic_reviews[0].match == "mismatch"
    assert any(f.category == "story" and f.severity == "warning" for f in report.findings)
    assert analyzers[0].comparison["purpose"] == "Establish the destination"
    assert analyzers[0].comparison["visual_description"] == "A wide view of the mountains"
    assert analyzers[0].unloaded is True


def test_semantic_sample_limit_supports_single_frame(tmp_path):
    _, timeline_path, plan_path = _write_inputs(tmp_path)
    timeline = ProjectDraftTimeline.model_validate_json(timeline_path.read_text())
    plan = VideoProjectPlan.model_validate_json(plan_path.read_text())

    from video_edit_agent.rendering.critique import _semantic_targets

    targets = _semantic_targets(
        timeline.clips * 3,
        plan,
        rendered_duration=30,
        sample_limit=1,
    )

    assert len(targets) == 1


def test_critique_reports_missing_audio_and_dark_frames(tmp_path, monkeypatch):
    _write_inputs(tmp_path)
    monkeypatch.setattr("video_edit_agent.rendering.critique.validate_artifact", lambda *_: False)
    monkeypatch.setattr("video_edit_agent.rendering.critique.shutil.which", lambda name: name)
    probe = {
        "format": {"duration": "10"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30/1",
            }
        ],
    }
    monkeypatch.setattr(
        "video_edit_agent.rendering.critique._run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, json.dumps(probe), ""),
    )
    dark = FrameQualitySample(
        timestamp_seconds=4,
        mean_luma=8,
        dark_pixel_ratio=0.8,
        bright_pixel_ratio=0,
        sharpness_variance=2,
    )
    monkeypatch.setattr("video_edit_agent.rendering.critique._sample_frames", lambda *_: [dark])

    report = critique_rendered_draft(tmp_path, sample_count=1)

    assert report.audio == AudioQualitySummary(has_audio=False)
    assert any(f.category == "audio" for f in report.findings)
    assert any(f.category == "visual" for f in report.findings)


def test_critique_rejects_unapproved_timeline(tmp_path, monkeypatch):
    _, timeline_path, _ = _write_inputs(tmp_path)
    timeline = ProjectDraftTimeline.model_validate_json(timeline_path.read_text())
    timeline.timeline_approved = False
    timeline_path.write_text(timeline.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr("video_edit_agent.rendering.critique.validate_artifact", lambda *_: False)

    with pytest.raises(DraftCritiqueError, match="fully reviewed and approved"):
        critique_rendered_draft(tmp_path)


def test_critique_rejects_paths_outside_project(tmp_path):
    _write_inputs(tmp_path)
    with pytest.raises(DraftCritiqueError, match="inside the project"):
        critique_rendered_draft(tmp_path, video_path=Path("../outside.mp4"))


def test_critique_rejects_invalid_sample_count(tmp_path):
    with pytest.raises(ValueError, match="between 1 and 24"):
        critique_rendered_draft(tmp_path, sample_count=25)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("30/1", 30), ("30000/1001", pytest.approx(30000 / 1001)), ("0/0", None), ("N/A", None)],
)


def test_quality_critique_findings_are_pure_and_do_not_require_ffmpeg(tmp_path):
    _, timeline_path, plan_path = _write_inputs(tmp_path)
    plan = VideoProjectPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    timeline = ProjectDraftTimeline.model_validate_json(timeline_path.read_text(encoding="utf-8"))

    findings = _quality_critique_findings(
        plan=plan,
        timeline=timeline,
        duration=8.0,
        width=640,
        height=360,
        frames=[_frame()],
        audio=AudioQualitySummary(has_audio=True, mean_volume_db=-18.0),
        semantic=False,
        semantic_reviews=[],
    )

    categories = {finding.category for finding in findings}
    assert "format" in categories
    assert "render" in categories
    assert all(isinstance(finding, CritiqueFinding) for finding in findings)


def test_semantic_critique_findings_are_pure_and_thresholded():
    findings = _semantic_critique_findings(
        [
            SemanticFrameReview(
                shot_id="shot-1",
                timestamp_seconds=2.5,
                match="mismatch",
                score=0.3,
                confidence=0.9,
                evidence="The rendered shot shows a city instead of the planned mountains.",
                suggestion="Review the selected source clip.",
            ),
            SemanticFrameReview(
                shot_id="shot-2",
                timestamp_seconds=5.0,
                match="partial",
                score=0.6,
                confidence=0.2,
                evidence="Low confidence comparison.",
                suggestion="Review if needed.",
            ),
        ]
    )

    assert len(findings) == 1
    assert findings[0].category == "story"
    assert findings[0].severity == "warning"
    assert "shot-1" in findings[0].message



def test_parse_frame_rate(value, expected):
    assert _parse_frame_rate(value) == expected


def test_critique_report_schema_rejects_invalid_frame_metrics():
    with pytest.raises(ValidationError):
        FrameQualitySample(
            timestamp_seconds=1,
            mean_luma=300,
            dark_pixel_ratio=0,
            bright_pixel_ratio=0,
            sharpness_variance=10,
        )


def test_critique_cli_writes_review_only_report(tmp_path, monkeypatch):
    video, timeline, plan = _write_inputs(tmp_path)
    report = DraftCritiqueReport(
        project_path=str(tmp_path.resolve()),
        rendered_video=video.relative_to(tmp_path).as_posix(),
        source_timeline=timeline.relative_to(tmp_path).as_posix(),
        source_plan=plan.relative_to(tmp_path).as_posix(),
        duration_seconds=10,
        width=1920,
        height=1080,
        frame_rate=30,
        video_codec="h264",
        audio_codec="aac",
        frames=[_frame()],
        audio=AudioQualitySummary(has_audio=True, mean_volume_db=-18, max_volume_db=-2),
        findings=[
            CritiqueFinding(
                severity="info",
                category="render",
                message="No threshold-based issues were detected.",
                evidence="Metadata and sampled media inspected.",
            )
        ],
    )
    monkeypatch.setattr(
        "video_edit_agent.cli.rendering.critique_rendered_draft",
        lambda *a, **k: report,
    )
    monkeypatch.setattr(
        "video_edit_agent.cli.rendering.register_artifact",
        lambda *a, **k: None,
    )
    result = runner.invoke(app, ["critique-draft", str(tmp_path), "--samples", "1"])

    assert result.exit_code == 0, result.output
    saved = tmp_path / ".video-agent" / "draft-critique.json"
    assert saved.is_file()
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["timeline_modified"] is False
    assert "Review Required" in result.output


def test_critique_cli_help_documents_sample_limit():
    result = runner.invoke(app, ["critique-draft", "--help"])
    assert result.exit_code == 0
    help_text = Text.from_ansi(result.output).plain
    assert "--samples" in help_text
    assert "--timeline" in help_text
    assert "--semantic" in help_text
    assert "--semantic-samples" in help_text
