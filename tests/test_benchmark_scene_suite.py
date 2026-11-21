import json
from pathlib import Path

import pytest

from scripts.benchmark_scene_suite import (
    inference_timing,
    latency_summary,
    load_manifest,
    validate_frame_output,
    validate_temporal_output,
)


def test_latency_summary_reports_distribution():
    result = latency_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["count"] == 5
    assert result["mean_seconds"] == 3.0
    assert result["p50_seconds"] == 3.0
    assert result["p95_seconds"] == 5.0


def test_manifest_validates_order_and_resolves_paths(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"frame")
    (tmp_path / "b.jpg").write_bytes(b"frame")
    manifest = tmp_path / "suite.json"
    manifest.write_text(json.dumps({"scenes": [{
        "scene_id": "one",
        "frames": [
            {"path": "a.jpg", "timestamp_seconds": 0},
            {"path": "b.jpg", "timestamp_seconds": 2.5},
        ],
    }]}))
    scenes = load_manifest(manifest)
    assert scenes[0]["frames"][1]["timestamp_seconds"] == 2.5
    assert scenes[0]["frames"][0]["path"] == (tmp_path / "a.jpg").resolve()


def test_manifest_rejects_non_monotonic_timestamps(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"frame")
    manifest = tmp_path / "suite.json"
    manifest.write_text(json.dumps({"scenes": [{
        "frames": [
            {"path": "a.jpg", "timestamp_seconds": 2},
            {"path": "a.jpg", "timestamp_seconds": 1},
        ],
    }]}))
    with pytest.raises(ValueError, match="ordered"):
        load_manifest(manifest)


def test_frame_and_temporal_outputs_use_production_schemas(tmp_path: Path):
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"frame")
    frame = {"path": frame_path, "timestamp_seconds": 1.25}
    observation = validate_frame_output({
        "description": "A performer juggles flaming torches.",
        "objects": ["performer", "torches"],
        "activities": ["juggling"],
        "confidence": 0.9,
        "shot_scale": "wide",
        "camera_angle": "eye_level",
        "composition": "unknown",
        "lighting": "daylight",
        "mood": "energetic",
        "visual_role": "action",
        "setting": "waterfront",
        "primary_subject": "performer",
    }, frame)
    assert observation["timestamp_seconds"] == 1.25
    assert observation["frame_path"] == str(frame_path)

    dynamics = validate_temporal_output({
        "temporal_summary": "Performer moves across the scene.",
        "action_progression": ["Moves left", "Turns toward audience"],
        "subject_motion": "crossing",
        "camera_motion": "static",
        "confidence": 0.8,
    })
    assert dynamics["camera_motion"] == "static"


def test_temporal_schema_rejects_invalid_enum():
    with pytest.raises(ValueError):
        validate_temporal_output({
            "temporal_summary": "A camera moves.",
            "action_progression": [],
            "subject_motion": "flying",
            "camera_motion": "static",
            "confidence": 0.8,
        })



def test_inference_timing_reports_measurable_phases():
    result = inference_timing(0.123456, 12.34567, '{"ok": true}')
    assert result == {
        "prompt_format_seconds": 0.1235,
        "generate_call_seconds": 12.3457,
        "output_characters": 12,
    }
