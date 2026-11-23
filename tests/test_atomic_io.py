"""Tests for crash-safe JSON artifact persistence."""

import json

import pytest
from pydantic import BaseModel

from video_edit_agent import atomic_io


class ExampleModel(BaseModel):
    title: str
    count: int


def test_write_json_atomic_serializes_models_and_creates_parent(tmp_path):
    destination = tmp_path / "nested" / "plan.json"

    atomic_io.write_json_atomic(destination, ExampleModel(title="旅行", count=3))

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "title": "旅行",
        "count": 3,
    }
    assert destination.read_text(encoding="utf-8").endswith("\n")


def test_write_json_atomic_accepts_json_values(tmp_path):
    destination = tmp_path / "report.json"

    atomic_io.write_json_atomic(destination, {"approved": False, "items": [1, 2]})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "approved": False,
        "items": [1, 2],
    }


def test_write_json_atomic_preserves_existing_file_on_replace_failure(tmp_path, monkeypatch):
    destination = tmp_path / "plan.json"
    destination.write_text('{"old": true}\n', encoding="utf-8")

    def fail_replace(source, target):
        raise OSError("simulated interrupted replacement")

    monkeypatch.setattr(atomic_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        atomic_io.write_json_atomic(destination, {"new": True})

    assert destination.read_text(encoding="utf-8") == '{"old": true}\n'
    assert list(tmp_path.glob(".plan.json.*.tmp")) == []


def test_write_json_atomic_preserves_existing_permissions(tmp_path):
    destination = tmp_path / "plan.json"
    destination.write_text("{}", encoding="utf-8")
    destination.chmod(0o640)

    atomic_io.write_json_atomic(destination, {"approved": True})

    assert destination.stat().st_mode & 0o777 == 0o640
