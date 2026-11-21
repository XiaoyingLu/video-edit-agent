import json
import sys
import types

import pytest

from video_edit_agent.understanding.provider import create_vision_analyzer
from video_edit_agent.understanding.qwen3_vl_analyzer import (
    DEFAULT_QWEN3_VL_MODEL,
    Qwen3VLMLXAnalyzer,
)
from video_edit_agent.understanding.visual_analyzer import VisionModelError


def _observation() -> dict:
    return {
        "description": "A performer moves a hoop beside the waterfront.",
        "objects": ["performer", "hoop", "waterfront"],
        "activities": ["performing"],
        "confidence": 0.95,
        "shot_scale": "wide",
        "camera_angle": "eye_level",
        "composition": "layered_depth",
        "lighting": "daylight",
        "mood": "energetic",
        "visual_role": "action",
        "setting": "waterfront promenade",
        "primary_subject": "performer",
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


def test_qwen3_vl_analyzes_frame_with_shared_observation_schema(monkeypatch, tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"image")
    calls = _install_fake_mlx(monkeypatch, [json.dumps(_observation())])

    result = Qwen3VLMLXAnalyzer().analyze_frame(frame)

    assert result["description"] == _observation()["description"]
    assert result["visual_role"] == "action"
    assert calls[0]["load"] == DEFAULT_QWEN3_VL_MODEL
    assert calls[1]["image"] == [str(frame)]
    assert calls[1]["temp"] == 0.0


def test_qwen3_vl_normalizes_list_valued_temporal_motion(monkeypatch, tmp_path):
    frames = []
    for index in range(3):
        frame = tmp_path / f"frame-{index}.jpg"
        frame.write_bytes(b"image")
        frames.append((float(index * 2), frame))
    temporal = {
        "temporal_summary": "A performer moves and interacts with a hoop.",
        "action_progression": ["Running", "Acrobatic movement with hoop", "Stationary"],
        "subject_motion": ["Running", "Acrobatic movement with hoop"],
        "camera_motion": ["Handheld", "Handheld"],
        "confidence": 0.95,
    }
    calls = _install_fake_mlx(monkeypatch, [json.dumps(temporal)])

    result = Qwen3VLMLXAnalyzer().analyze_scene(frames)

    assert result["subject_motion"] == "mixed"
    assert result["camera_motion"] == "handheld"
    assert result["action_progression"] == temporal["action_progression"]
    assert calls[1]["image"] == [str(frame) for _, frame in frames]
    assert "0.00s, 2.00s, 4.00s" in calls[1]["prompt"]


def test_qwen3_vl_retries_invalid_temporal_output(monkeypatch, tmp_path):
    frames = []
    for index in range(2):
        frame = tmp_path / f"frame-{index}.jpg"
        frame.write_bytes(b"image")
        frames.append((float(index), frame))
    valid = {
        "temporal_summary": "A cyclist crosses the frame.",
        "action_progression": ["The cyclist enters", "The cyclist crosses"],
        "subject_motion": "crossing",
        "camera_motion": "static",
        "confidence": 0.8,
    }
    calls = _install_fake_mlx(
        monkeypatch,
        [
            json.dumps(
                {"subject_motion": ["unclassified"], "camera_motion": "unknown", "confidence": 1.5}
            ),
            json.dumps(valid),
        ],
    )

    result = Qwen3VLMLXAnalyzer().analyze_scene(frames)

    assert result["subject_motion"] == "crossing"
    assert len([call for call in calls if "prompt" in call]) == 2
    assert calls[-1]["max_tokens"] >= 2048


def test_qwen3_vl_reports_missing_optional_dependency(monkeypatch, tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"image")
    monkeypatch.setitem(sys.modules, "mlx_vlm", None)

    with pytest.raises(VisionModelError, match="requires mlx-vlm"):
        Qwen3VLMLXAnalyzer().analyze_frame(frame)


def test_provider_factory_selects_qwen3_vl_without_loading_runtime():
    analyzer = create_vision_analyzer("qwen3-vl")

    assert isinstance(analyzer, Qwen3VLMLXAnalyzer)
    assert analyzer.model_id == DEFAULT_QWEN3_VL_MODEL
