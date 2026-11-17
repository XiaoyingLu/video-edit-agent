from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from travel_video_agent.storage_cleanup import cleanup_generated_storage


def _write(path: Path, content: bytes = b"frame", *, age_days: float = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    timestamp = time.time() - age_days * 86400
    os.utime(path, (timestamp, timestamp))
    return path


def test_cleanup_dry_run_reports_without_deleting(tmp_path: Path) -> None:
    frame = _write(tmp_path / ".video-agent/frames/a/frame-01.jpg", age_days=20)
    report = cleanup_generated_storage(tmp_path, max_age_days=7, dry_run=True)

    assert report.scanned_files == 1
    assert report.candidate_files == 1
    assert report.removed_files == 0
    assert report.dry_run is True
    assert frame.exists()


def test_cleanup_removes_old_frames_but_preserves_durable_artifacts(tmp_path: Path) -> None:
    old_frame = _write(tmp_path / ".video-agent/frames/a/frame-01.jpg", age_days=20)
    recent_frame = _write(tmp_path / ".video-agent/frames/b/frame-01.jpg")
    manifest = _write(tmp_path / ".video-agent/artifact-manifest.json", b"{}")
    render = _write(tmp_path / ".video-agent/rendered-draft.mp4", b"mp4")
    model = _write(tmp_path / ".video-agent/models/whisper/model.bin", b"weights")

    report = cleanup_generated_storage(tmp_path, max_age_days=7)

    assert report.removed_files == 1
    assert not old_frame.exists()
    assert recent_frame.exists()
    assert manifest.exists()
    assert render.exists()
    assert model.exists()


def test_cleanup_enforces_size_budget_oldest_first(tmp_path: Path) -> None:
    oldest = _write(tmp_path / ".video-agent/frames/a.jpg", b"a" * 10, age_days=2)
    newest = _write(tmp_path / ".video-agent/highlight-frames/b.jpg", b"b" * 10)
    report = cleanup_generated_storage(tmp_path, max_age_days=30, max_cache_gb=1e-8)

    assert report.removed_files == 1
    assert not oldest.exists()
    assert newest.exists()


def test_keep_artifacts_protects_understanding_referenced_frames(tmp_path: Path) -> None:
    frame = _write(tmp_path / ".video-agent/frames/a/frame-01.jpg", age_days=30)
    understanding = {
        "scenes": [{"observations": [{"frame_path": ".video-agent/frames/a/frame-01.jpg"}]}]
    }
    (tmp_path / ".video-agent/understanding.json").write_text(json.dumps(understanding))

    report = cleanup_generated_storage(tmp_path, max_age_days=1, keep_artifacts=True, dry_run=False)

    assert report.scanned_files == 1
    assert report.candidate_files == 0
    assert frame.exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_age_days": -1}, "max_age_days"),
        ({"max_cache_gb": 0}, "max_cache_gb"),
        ({"max_age_days": float("nan")}, "max_age_days"),
        ({"max_cache_gb": float("inf")}, "max_cache_gb"),
    ],
)
def test_cleanup_rejects_invalid_limits(tmp_path: Path, kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cleanup_generated_storage(tmp_path, **kwargs)


def test_cleanup_rejects_missing_project(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        cleanup_generated_storage(tmp_path / "missing")
