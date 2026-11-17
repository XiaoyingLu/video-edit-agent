import json
import os

import pytest

from travel_video_agent.artifact_tracking import (
    StaleArtifactError,
    register_artifact,
    validate_all_artifacts,
    validate_artifact,
)


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_registered_artifact_is_valid_until_source_content_changes(tmp_path):
    source = _write(tmp_path / "trip.mp4", "original video bytes")
    artifact = _write(tmp_path / ".video-agent" / "understanding.json", '{"scenes": []}')

    register_artifact(tmp_path, artifact, config={"model": "qwen2.5vl:3b"})
    assert validate_artifact(tmp_path, artifact) is True

    source.write_text("changed video bytes", encoding="utf-8")
    os.utime(source, None)

    with pytest.raises(StaleArtifactError, match="Source video"):
        validate_artifact(tmp_path, artifact)


def test_dependency_change_invalidates_downstream_artifact(tmp_path):
    _write(tmp_path / "trip.mp4", "video")
    upstream = _write(tmp_path / ".video-agent" / "understanding.json", '{"scene_count": 1}')
    downstream = _write(tmp_path / ".video-agent" / "highlights.json", '{"candidate_count": 1}')

    register_artifact(tmp_path, upstream)
    register_artifact(tmp_path, downstream, dependencies=[".video-agent/understanding.json"])
    assert validate_artifact(tmp_path, downstream) is True

    upstream.write_text('{"scene_count": 2}', encoding="utf-8")
    with pytest.raises(StaleArtifactError, match="modified after it was generated"):
        validate_artifact(tmp_path, downstream)


def test_new_source_video_invalidates_existing_artifacts(tmp_path):
    _write(tmp_path / "trip-a.mp4", "video a")
    artifact = _write(tmp_path / ".video-agent" / "analysis.json", '{"video_count": 1}')
    register_artifact(tmp_path, artifact)

    _write(tmp_path / "trip-b.mp4", "video b")
    stale = validate_all_artifacts(tmp_path)
    assert len(stale) == 1
    assert "Source video set changed" in stale[0]


def test_legacy_artifact_without_manifest_entry_remains_usable(tmp_path):
    artifact = _write(tmp_path / ".video-agent" / "legacy.json", json.dumps({"ok": True}))
    assert validate_artifact(tmp_path, artifact) is False


def test_artifact_tampering_is_detected(tmp_path):
    _write(tmp_path / "trip.mp4", "video")
    artifact = _write(tmp_path / ".video-agent" / "analysis.json", '{"video_count": 1}')
    register_artifact(tmp_path, artifact)
    artifact.write_text('{"video_count": 99}', encoding="utf-8")

    with pytest.raises(StaleArtifactError, match="was modified"):
        validate_artifact(tmp_path, artifact)
