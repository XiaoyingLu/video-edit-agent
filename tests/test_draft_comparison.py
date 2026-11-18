import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from travel_video_agent.cli import app
from travel_video_agent.rendering.comparison import (
    DraftComparisonError,
    DraftComparisonReport,
    compare_rendered_drafts,
    render_comparison_html,
)
from travel_video_agent.rendering.critique import AudioQualitySummary

runner = CliRunner()


def _media(root: Path) -> tuple[Path, Path]:
    artifact = root / ".video-agent"
    artifact.mkdir(parents=True, exist_ok=True)
    before = artifact / "before.mp4"
    after = artifact / "after.mp4"
    before.write_bytes(b"before")
    after.write_bytes(b"after")
    return before, after


def _audio(mean: float | None = -18, *, has_audio: bool = True):
    return AudioQualitySummary(
        has_audio=has_audio,
        mean_volume_db=mean,
        max_volume_db=-2 if has_audio else None,
        silence_intervals=[(2, 3)] if has_audio else [],
    )


def _mock_tools(monkeypatch, before, after):
    metadata = {
        str(before): {
            "format": {"duration": "10"},
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
        },
        str(after): {
            "format": {"duration": "8"},
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
        },
    }
    monkeypatch.setattr(
        "travel_video_agent.rendering.comparison.validate_artifact", lambda *_: False
    )
    monkeypatch.setattr(
        "travel_video_agent.rendering.comparison.shutil.which", lambda name: name
    )
    monkeypatch.setattr(
        "travel_video_agent.rendering.comparison._analyze_audio",
        lambda _ffmpeg, video, _has_audio: _audio(-18 if video == before else -16),
    )

    def fake_run(command, **kwargs):
        if command[0] == "ffprobe":
            return subprocess.CompletedProcess(command, 0, json.dumps(metadata[command[-1]]), "")
        value = 90 if "before-" in command[-1] else 120
        image = np.full((24, 32, 3), value, dtype=np.uint8)
        assert cv2.imwrite(command[-1], image)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("travel_video_agent.rendering.comparison._run", fake_run)


def test_comparison_pairs_samples_by_normalized_position_and_reports_deltas(tmp_path, monkeypatch):
    before, after = _media(tmp_path)
    _mock_tools(monkeypatch, before, after)

    report = compare_rendered_drafts(
        tmp_path,
        before_video=before,
        after_video=after,
        sample_count=2,
    )

    assert report.before.duration_seconds == 10
    assert report.after.duration_seconds == 8
    assert report.duration_delta_seconds == -2
    assert report.mean_volume_delta_db == 2
    assert report.frames[0].normalized_position == pytest.approx(1 / 3)
    assert report.frames[0].before.timestamp_seconds == pytest.approx(10 / 3)
    assert report.frames[0].after.timestamp_seconds == pytest.approx(8 / 3)
    assert report.frames[0].mean_luma_delta > 0
    assert report.timeline_modified is False
    assert report.rendered_videos_modified is False
    serialized = json.dumps(report.model_dump(mode="json"))
    restored = DraftComparisonReport.model_validate_json(serialized)
    assert restored.frames[0].before_image_data_uri == ""


def test_comparison_rejects_same_video_and_paths_outside_project(tmp_path, monkeypatch):
    before, _ = _media(tmp_path)
    monkeypatch.setattr(
        "travel_video_agent.rendering.comparison.validate_artifact", lambda *_: False
    )
    with pytest.raises(DraftComparisonError, match="different files"):
        compare_rendered_drafts(tmp_path, before_video=before, after_video=before)
    with pytest.raises(DraftComparisonError, match="inside the project"):
        compare_rendered_drafts(
            tmp_path,
            before_video=Path("../before.mp4"),
            after_video=before,
        )


def test_comparison_rejects_invalid_sample_count(tmp_path):
    before, after = _media(tmp_path)
    with pytest.raises(ValueError, match="between 1 and 16"):
        compare_rendered_drafts(
            tmp_path,
            before_video=before,
            after_video=after,
            sample_count=17,
        )


def test_html_contains_paired_visual_evidence_and_caveat(tmp_path, monkeypatch):
    before, after = _media(tmp_path)
    _mock_tools(monkeypatch, before, after)
    report = compare_rendered_drafts(
        tmp_path, before_video=before, after_video=after, sample_count=1
    )

    page = render_comparison_html(report)

    assert "Before · 5.00s" in page
    assert "After · 4.00s" in page
    assert "data:image/jpeg;base64," in page
    assert "not a score of creative quality" in page


def test_cli_writes_json_and_standalone_html(tmp_path, monkeypatch):
    before, after = _media(tmp_path)
    _mock_tools(monkeypatch, before, after)
    monkeypatch.setattr(
        "travel_video_agent.cli.register_artifact", lambda *_args, **_kwargs: None
    )

    result = runner.invoke(
        app,
        [
            "compare-drafts",
            str(tmp_path),
            "--before",
            str(before),
            "--after",
            str(after),
            "--samples",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    json_path = tmp_path / ".video-agent" / "draft-comparison.json"
    html_path = tmp_path / ".video-agent" / "draft-comparison.html"
    assert json_path.is_file()
    assert html_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["duration_delta_seconds"] == -2
    assert "data:image" not in json_path.read_text(encoding="utf-8")
    assert "data:image/jpeg;base64," in html_path.read_text(encoding="utf-8")


def test_cli_help_documents_comparison_inputs():
    result = runner.invoke(app, ["compare-drafts", "--help"])
    assert result.exit_code == 0
    assert "--before" in result.output
    assert "--after" in result.output
    assert "--samples" in result.output
