import json
from pathlib import Path

import pytest

from scripts import benchmark_scene_suite
from scripts.benchmark_scene_suite import (
    inference_timing,
    latency_summary,
    load_manifest,
    run_smol_suite,
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
    manifest.write_text(
        json.dumps(
            {
                "scenes": [
                    {
                        "scene_id": "one",
                        "frames": [
                            {"path": "a.jpg", "timestamp_seconds": 0},
                            {"path": "b.jpg", "timestamp_seconds": 2.5},
                        ],
                    }
                ]
            }
        )
    )
    scenes = load_manifest(manifest)
    assert scenes[0]["frames"][1]["timestamp_seconds"] == 2.5
    assert scenes[0]["frames"][0]["path"] == (tmp_path / "a.jpg").resolve()


def test_manifest_rejects_non_monotonic_timestamps(tmp_path: Path):
    (tmp_path / "a.jpg").write_bytes(b"frame")
    manifest = tmp_path / "suite.json"
    manifest.write_text(
        json.dumps(
            {
                "scenes": [
                    {
                        "frames": [
                            {"path": "a.jpg", "timestamp_seconds": 2},
                            {"path": "a.jpg", "timestamp_seconds": 1},
                        ],
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="ordered"):
        load_manifest(manifest)


def test_frame_and_temporal_outputs_use_production_schemas(tmp_path: Path):
    frame_path = tmp_path / "frame.jpg"
    frame_path.write_bytes(b"frame")
    frame = {"path": frame_path, "timestamp_seconds": 1.25}
    observation = validate_frame_output(
        {
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
        },
        frame,
    )
    assert observation["timestamp_seconds"] == 1.25
    assert observation["frame_path"] == str(frame_path)

    dynamics = validate_temporal_output(
        {
            "temporal_summary": "Performer moves across the scene.",
            "action_progression": ["Moves left", "Turns toward audience"],
            "subject_motion": "crossing",
            "camera_motion": "static",
            "confidence": 0.8,
        }
    )
    assert dynamics["camera_motion"] == "static"


def test_temporal_schema_rejects_invalid_enum():
    with pytest.raises(ValueError):
        validate_temporal_output(
            {
                "temporal_summary": "A camera moves.",
                "action_progression": [],
                "subject_motion": "flying",
                "camera_motion": "static",
                "confidence": 0.8,
            }
        )


def test_normalize_temporal_payload_collapses_qwen_enum_arrays():
    raw = {
        "temporal_summary": "A fire performer is followed by a hoop performer.",
        "action_progression": ["Runs", "Performs acrobatics with hoop", "Stops"],
        "subject_motion": ["Running", "Acrobatic movement with hoop", "Stationary"],
        "camera_motion": ["Handheld", "Handheld"],
        "confidence": 0.95,
    }
    normalized, changed = benchmark_scene_suite.normalize_temporal_payload(raw)

    assert normalized["subject_motion"] == "mixed"
    assert normalized["camera_motion"] == "handheld"
    assert set(changed) == {"subject_motion", "camera_motion"}
    assert raw["subject_motion"] == ["Running", "Acrobatic movement with hoop", "Stationary"]
    assert validate_temporal_output(normalized)["camera_motion"] == "handheld"


def test_normalize_temporal_payload_preserves_known_scalar_enums():
    raw = {
        "temporal_summary": "A performer remains in place.",
        "action_progression": [],
        "subject_motion": "stationary",
        "camera_motion": "static",
        "confidence": 0.8,
    }
    normalized, changed = benchmark_scene_suite.normalize_temporal_payload(raw)

    assert normalized == raw
    assert changed == []


def test_inference_timing_reports_measurable_phases():
    result = inference_timing(0.123456, 12.34567, '{"ok": true}')
    assert result == {
        "prompt_format_seconds": 0.1235,
        "generate_call_seconds": 12.3457,
        "output_characters": 12,
    }


def test_smol_suite_runs_without_qwen_timing_metrics(monkeypatch, tmp_path: Path):
    class FakeAnalyzer:
        def __init__(self, model: str):
            self.model = model

        def _load(self):
            return None

        def analyze_frame(self, frame_path: Path):
            return {
                "description": "A performer juggles torches.",
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
            }

        def analyze_scene(self, frames):
            return {
                "temporal_summary": "A performer juggles for an audience.",
                "action_progression": ["Juggles torches"],
                "subject_motion": "stationary",
                "camera_motion": "static",
                "confidence": 0.9,
            }

        def unload_model(self):
            return None

    monkeypatch.setattr(benchmark_scene_suite, "SmolVLM2MLXAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(benchmark_scene_suite, "_mlx_memory", lambda: {})

    frames = [
        {"path": tmp_path / "one.jpg", "timestamp_seconds": 0.0},
        {"path": tmp_path / "two.jpg", "timestamp_seconds": 1.0},
    ]
    result = run_smol_suite(
        [{"scene_id": "one", "frames": frames, "review_notes": "", "expected_events": []}],
        "fake-model",
    )

    assert result["status"] == "completed"
    assert result["frame_schema_valid_count"] == 2
    assert result["temporal_schema_valid_count"] == 1
    assert result["scenes"][0]["temporal"]["schema_validation"]["valid"] is True
    assert result["scenes"][0]["temporal"]["normalization_applied"] == []
    assert "inference_timing" not in result["scenes"][0]["frames"][0]
