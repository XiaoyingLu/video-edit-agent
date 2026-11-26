from pathlib import Path
from types import SimpleNamespace

import pytest

from video_edit_agent import runtime_capabilities as capabilities


def test_probe_builds_tool_and_skill_allowlists_from_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(
        capabilities,
        "discover_videos",
        lambda _root: [tmp_path / "clip.mp4"],
    )
    monkeypatch.setattr(
        capabilities.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"ffmpeg", "ffprobe"} else None,
    )

    def fake_run(command, **_kwargs):
        output = (
            "Filters:\n ..C scale V->V scale video\n ... pad V->V pad video\n"
            " ... setsar V->V set sample aspect\n ... fps V->V frame rate\n"
            " ... format V->V format\n ... trim V->V trim\n"
            " ... setpts V->V set timestamps\n ... concat N->N concat\n"
            " ... aresample A->A resample audio\n ... aformat A->A audio format\n"
            " ... anullsrc |->A null audio\n ... subtitles V->V render subtitles\n"
            " ... drawtext V->V draw text\n"
            if command[-1] == "-filters"
            else "Encoders:\n V....D libx264 H.264\n A....D aac AAC\n"
        )
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(capabilities.subprocess, "run", fake_run)
    monkeypatch.setattr(
        capabilities,
        "_available_fonts",
        lambda: frozenset({"Lato", "DejaVu Sans"}),
    )
    monkeypatch.setattr(
        capabilities.importlib.util,
        "find_spec",
        lambda name: object() if name in {"cv2", "faster_whisper"} else None,
    )
    monkeypatch.setattr(
        capabilities,
        "get_setting",
        lambda name: {
            "VISION_PROVIDER": "ollama",
            "SOUNDSTRIPE_API_KEY": "test-soundstripe-key",
            "STABILITY_API_KEY": "",
        }.get(name),
    )

    report = capabilities.probe_runtime_capabilities(tmp_path)

    assert report.has("inspect_media")
    assert report.has("scene_understanding")
    assert report.has("visual_analysis")
    assert report.has("transcription")
    assert report.has("render_video")
    assert report.has("render_subtitles")
    assert report.has("render_titles")
    assert report.has("search_soundtrack_catalog")
    assert not report.has("generate_soundtrack")
    assert "SOUNDSTRIPE_API_KEY" in report.external_keys
    assert "STABILITY_API_KEY" not in report.external_keys
    assert "test-soundstripe-key" not in str(report.prompt_context())


def test_require_fails_closed_with_specific_missing_capability():
    report = capabilities.CapabilityReport(
        project_root="/project",
        unavailable={"transcription": "faster-whisper is not installed."},
    )

    with pytest.raises(
        capabilities.RuntimeCapabilityError,
        match="transcription: faster-whisper is not installed",
    ):
        report.require("transcription")


def test_probe_reports_missing_media_and_ffmpeg_capabilities(monkeypatch, tmp_path):
    monkeypatch.setattr(capabilities, "discover_videos", lambda _root: [])
    monkeypatch.setattr(capabilities.shutil, "which", lambda _name: None)
    monkeypatch.setattr(capabilities, "_available_fonts", lambda: frozenset())
    monkeypatch.setattr(capabilities.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(capabilities, "get_setting", lambda _name: None)

    report = capabilities.probe_runtime_capabilities(tmp_path)

    assert not report.has("inspect_media")
    assert not report.has("render_video")
    assert not report.has("render_subtitles")
    with pytest.raises(capabilities.RuntimeCapabilityError, match="No supported video files"):
        report.require("inspect_media")


def test_discover_videos_rejects_missing_project_without_leaking_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(
        capabilities,
        "discover_videos",
        lambda _root: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    monkeypatch.setattr(capabilities.shutil, "which", lambda _name: None)
    monkeypatch.setattr(capabilities, "_available_fonts", lambda: frozenset())
    monkeypatch.setattr(capabilities.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(capabilities, "get_setting", lambda _name: None)

    report = capabilities.probe_runtime_capabilities(Path(tmp_path) / "missing")

    assert report.media_files == ()
