import json
import sys
import types

import pytest

from travel_video_agent.understanding.smolvlm2_analyzer import SmolVLM2MLXAnalyzer
from travel_video_agent.understanding.visual_analyzer import VisionModelError


def _observation() -> dict:
    return {
        "description": "A hiker walks beside a mountain lake.",
        "objects": ["hiker", "lake", "mountains"],
        "activities": ["walking"],
        "confidence": 0.9,
        "shot_scale": "wide",
        "camera_angle": "eye_level",
        "composition": "layered_depth",
        "lighting": "daylight",
        "mood": "serene",
        "visual_role": "establishing",
        "setting": "mountain lake",
        "primary_subject": "hiker",
    }


def _install_fake_mlx(monkeypatch, generated: list[str]) -> list[dict]:
    calls = []

    def load(model_id):
        calls.append({"load": model_id})
        return object(), object()

    def generate(model, processor, prompt, images, **kwargs):
        calls.append({"prompt": prompt, "image": images, **kwargs})
        return generated.pop(0)

    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm",
        types.SimpleNamespace(load=load, generate=generate),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.prompt_utils",
        types.SimpleNamespace(
            apply_chat_template=lambda processor, config, prompt, num_images: prompt
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.utils",
        types.SimpleNamespace(load_config=lambda model_id: {"model": model_id}),
    )
    return calls


def test_smolvlm2_mlx_analyzes_frame_with_shared_observation_schema(monkeypatch, tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"image")
    calls = _install_fake_mlx(monkeypatch, [json.dumps(_observation())])

    result = SmolVLM2MLXAnalyzer().analyze_frame(frame)

    assert result["description"] == _observation()["description"]
    assert result["visual_role"] == "establishing"
    assert calls[0]["load"] == "mlx-community/SmolVLM2-2.2B-Instruct-mlx"
    assert calls[1]["image"] == [str(frame)]
    assert calls[1]["temp"] == 0.0


def test_smolvlm2_mlx_analyzes_ordered_frames_for_temporal_dynamics(monkeypatch, tmp_path):
    frames = []
    for index in range(2):
        frame = tmp_path / f"frame-{index}.jpg"
        frame.write_bytes(b"image")
        frames.append((float(index * 2), frame))
    temporal = {
        "temporal_summary": "A hiker walks toward the viewpoint.",
        "action_progression": ["The hiker enters", "The hiker approaches the viewpoint"],
        "subject_motion": "approaching",
        "camera_motion": "static",
        "confidence": 0.85,
    }
    calls = _install_fake_mlx(monkeypatch, [json.dumps(temporal)])

    result = SmolVLM2MLXAnalyzer().analyze_scene(frames)

    assert result["subject_motion"] == "approaching"
    assert result["camera_motion"] == "static"
    assert calls[1]["image"] == [str(frames[0][1]), str(frames[1][1])]
    assert "0.00s, 2.00s" in calls[1]["prompt"]


def test_smolvlm2_mlx_rejects_non_json_output(monkeypatch, tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"image")
    calls = _install_fake_mlx(monkeypatch, ["I cannot analyze this image.", ""])

    with pytest.raises(VisionModelError, match="invalid JSON") as error:
        SmolVLM2MLXAnalyzer().analyze_frame(frame)

    assert "First attempt: The vision model returned invalid JSON" in str(error.value)
    assert "Retry attempt: The vision model returned empty or non-text output" in str(error.value)
    assert isinstance(error.value.__cause__, VisionModelError)
    assert "empty or non-text output" in str(error.value.__cause__)
    assert len([call for call in calls if "prompt" in call]) == 2


def test_smolvlm2_mlx_reports_missing_optional_dependency(monkeypatch, tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"image")
    monkeypatch.setitem(sys.modules, "mlx_vlm", None)

    with pytest.raises(VisionModelError, match="requires mlx-vlm"):
        SmolVLM2MLXAnalyzer().analyze_frame(frame)


def test_provider_factory_selects_mlx_without_loading_optional_runtime():
    from travel_video_agent.understanding.provider import create_vision_analyzer

    analyzer = create_vision_analyzer("mlx", "test/model")

    assert isinstance(analyzer, SmolVLM2MLXAnalyzer)
    assert analyzer.model_id == "test/model"


def test_smolvlm2_mlx_retries_incomplete_frame_json(monkeypatch, tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"image")
    calls = _install_fake_mlx(
        monkeypatch, ['{"description": "A parade", "objects": [', json.dumps(_observation())]
    )

    result = SmolVLM2MLXAnalyzer().analyze_frame(frame)

    assert result["description"] == _observation()["description"]
    assert len([call for call in calls if "prompt" in call]) == 2
    assert calls[1]["max_tokens"] == 1024
    assert calls[-1]["max_tokens"] >= 2048



def test_smolvlm2_mlx_normalizes_scalar_frame_arrays(monkeypatch, tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"image")
    observation = _observation()
    observation["objects"] = "unknown"
    observation["activities"] = "walking"
    _install_fake_mlx(monkeypatch, [json.dumps(observation)])

    result = SmolVLM2MLXAnalyzer().analyze_frame(frame)

    assert result["objects"] == []
    assert result["activities"] == ["walking"]


def test_smolvlm2_mlx_normalizes_scalar_temporal_progression(monkeypatch, tmp_path):
    frames = []
    for index in range(2):
        frame = tmp_path / f"frame-{index}.jpg"
        frame.write_bytes(b"image")
        frames.append((float(index), frame))
    temporal = {
        "temporal_summary": "A parade moves through the street.",
        "action_progression": "unknown",
        "subject_motion": "mixed",
        "camera_motion": "unknown",
        "confidence": 0.7,
    }
    _install_fake_mlx(monkeypatch, [json.dumps(temporal)])

    result = SmolVLM2MLXAnalyzer().analyze_scene(frames)

    assert result["action_progression"] == []
    assert result["subject_motion"] == "mixed"
