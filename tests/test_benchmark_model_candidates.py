"""Tests for the isolated video model candidate benchmark."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_model_candidates.py"
SPEC = importlib.util.spec_from_file_location("benchmark_model_candidates", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_json_object_validation_accepts_dict_and_fenced_json():
    assert benchmark._is_json_object({"description": "A mountain"})
    assert benchmark._is_json_object('{"description": "A mountain"}')
    assert benchmark._is_json_object('```json\n{"description": "A mountain"}\n```')


def test_json_object_validation_rejects_non_object_and_invalid_json():
    assert not benchmark._is_json_object("A mountain lake.")
    assert not benchmark._is_json_object("[1, 2]")
    assert not benchmark._is_json_object('{"description":')


def test_benchmark_candidates_are_opt_in_and_defaults_are_documented():
    assert benchmark.DEFAULT_QWEN_MODEL == "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    assert benchmark.DEFAULT_VIDEOCHAT_MODEL == "MCG-NJU/VideoChat3-4B"
    assert benchmark.DEFAULT_SMOL_MODEL == "mlx-community/SmolVLM2-2.2B-Instruct-mlx"
