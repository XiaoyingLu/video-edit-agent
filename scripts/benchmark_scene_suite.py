"""Multi-scene quality and stability benchmark for MLX vision models.

Manifest format:
{
  "scenes": [
    {"scene_id": "scene-001", "frames": [
      {"path": "frames/001.jpg", "timestamp_seconds": 0.0},
      {"path": "frames/002.jpg", "timestamp_seconds": 3.2}
    ], "review_notes": "Performer moves from left to right"}
  ]
}
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from travel_video_agent.understanding.schemas import FrameObservation, SceneDynamics
from travel_video_agent.understanding.smolvlm2_analyzer import (
    _FRAME_PROMPT,
    _TEMPORAL_PROMPT,
    _json_object,
    SmolVLM2MLXAnalyzer,
)
from travel_video_agent.understanding.visual_analyzer import _parse_observation

DEFAULT_SMOL_MODEL = "mlx-community/SmolVLM2-2.2B-Instruct-mlx"
DEFAULT_QWEN_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("scenes"), list):
        raise ValueError("Manifest must contain a 'scenes' array.")
    scenes = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(data["scenes"]):
        if not isinstance(raw, dict):
            raise ValueError(f"Scene {index} must be an object.")
        scene_id = str(raw.get("scene_id") or f"scene-{index + 1:03d}")
        if scene_id in seen_ids:
            raise ValueError(f"Duplicate scene_id: {scene_id}")
        seen_ids.add(scene_id)
        raw_frames = raw.get("frames")
        if not isinstance(raw_frames, list) or len(raw_frames) < 2:
            raise ValueError(f"{scene_id} must contain at least two frames.")
        frames = []
        previous_timestamp = -1.0
        for frame in raw_frames:
            if not isinstance(frame, dict) or "path" not in frame:
                raise ValueError(f"{scene_id} frames require path and timestamp_seconds.")
            frame_path = (path.parent / frame["path"]).expanduser().resolve()
            timestamp = float(frame["timestamp_seconds"])
            if timestamp < 0 or timestamp < previous_timestamp:
                raise ValueError(f"{scene_id} timestamps must be non-negative and ordered.")
            if not frame_path.is_file():
                raise ValueError(f"Frame does not exist: {frame_path}")
            previous_timestamp = timestamp
            frames.append({"path": frame_path, "timestamp_seconds": timestamp})
        scenes.append({
            "scene_id": scene_id,
            "frames": frames,
            "review_notes": str(raw.get("review_notes", "")),
            "expected_events": raw.get("expected_events", []),
        })
    if not scenes:
        raise ValueError("Manifest must contain at least one scene.")
    return scenes


def _json_payload(raw: str | dict[str, Any]) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else _json_object(raw, label="benchmark output")


def validate_frame_output(raw: str | dict[str, Any], frame: dict[str, Any]) -> dict[str, Any]:
    payload = _json_payload(raw)
    payload["timestamp_seconds"] = frame["timestamp_seconds"]
    payload["frame_path"] = str(frame["path"])
    normalized = _parse_observation(json.dumps(payload), frame["path"])
    normalized["timestamp_seconds"] = frame["timestamp_seconds"]
    return FrameObservation.model_validate(normalized).model_dump()


def validate_temporal_output(raw: str | dict[str, Any]) -> dict[str, Any]:
    return SceneDynamics.model_validate(_json_payload(raw)).model_dump()


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean_seconds": 0.0, "p50_seconds": 0.0, "p95_seconds": 0.0}
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * 0.95 + 0.5)))
    return {
        "count": len(values),
        "mean_seconds": round(statistics.mean(values), 3),
        "p50_seconds": round(statistics.median(values), 3),
        "p95_seconds": round(ordered[p95_index], 3),
    }


def _mlx_memory() -> dict[str, float]:
    try:
        import mlx.core as mx
        return {
            "active_mb": round(mx.get_active_memory() / 1024**2, 1),
            "cache_mb": round(mx.get_cache_memory() / 1024**2, 1),
            "peak_mb": round(mx.get_peak_memory() / 1024**2, 1),
        }
    except (ImportError, AttributeError):
        return {}


def run_smol_suite(scenes: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    analyzer = SmolVLM2MLXAnalyzer(model=model_id)
    scene_results = []
    frame_latencies: list[float] = []
    temporal_latencies: list[float] = []
    load_started = time.perf_counter()
    try:
        analyzer._load()
        load_seconds = time.perf_counter() - load_started
        for scene in scenes:
            frame_results = []
            scene_started = time.perf_counter()
            for frame in scene["frames"]:
                started = time.perf_counter()
                try:
                    observation = analyzer.analyze_frame(frame["path"])
                    observation["timestamp_seconds"] = frame["timestamp_seconds"]
                    observation["frame_path"] = str(frame["path"])
                    parsed = FrameObservation.model_validate(observation).model_dump()
                    validation = {"valid": True, "errors": []}
                except Exception as exc:
                    parsed = {"error": str(exc)}
                    validation = {"valid": False, "errors": [str(exc)]}
                elapsed = time.perf_counter() - started
                frame_latencies.append(elapsed)
                frame_results.append({
                    "frame": str(frame["path"]),
                    "timestamp_seconds": frame["timestamp_seconds"],
                    "latency_seconds": round(elapsed, 3),
                    "schema_validation": validation,
                    "observation": parsed,
                })
            temporal_started = time.perf_counter()
            temporal = analyzer.analyze_scene([
                (frame["timestamp_seconds"], frame["path"]) for frame in scene["frames"]
            ])
            temporal_elapsed = time.perf_counter() - temporal_started
            temporal_latencies.append(temporal_elapsed)
            try:
                temporal = validate_temporal_output(temporal)
                temporal_validation = {"valid": True, "errors": []}
            except Exception as exc:
                temporal_validation = {"valid": False, "errors": [str(exc)]}
            scene_results.append({
                "scene_id": scene["scene_id"],
                "elapsed_seconds": round(time.perf_counter() - scene_started, 3),
                "review_notes": scene["review_notes"],
                "expected_events": scene["expected_events"],
                "frames": frame_results,
                "temporal": {
                    "latency_seconds": round(temporal_elapsed, 3),
                    "schema_validation": temporal_validation,
                    "output": temporal,
                },
                "memory_after_scene": _mlx_memory(),
            })
        return _suite_result("smolvlm2", model_id, load_seconds, scene_results,
                             frame_latencies, temporal_latencies)
    finally:
        analyzer.unload_model()


def run_qwen_suite(scenes: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    import mlx.core as mx

    load_started = time.perf_counter()
    model, processor = load(model_id)
    config = load_config(model_id)
    load_seconds = time.perf_counter() - load_started
    mx.reset_peak_memory()

    def infer(prompt: str, paths: list[Path]) -> str:
        formatted = apply_chat_template(processor, config, prompt, num_images=len(paths))
        result = generate(model, processor, formatted, [str(p) for p in paths],
                          max_tokens=1024, temp=0.0)
        if isinstance(result, str):
            return result
        text = getattr(result, "text", None)
        if isinstance(text, str):
            return text
        raise RuntimeError(f"Unsupported MLX-VLM result: {type(result).__name__}")

    scene_results = []
    frame_latencies: list[float] = []
    temporal_latencies: list[float] = []
    try:
        for scene in scenes:
            frame_results = []
            scene_started = time.perf_counter()
            for frame in scene["frames"]:
                started = time.perf_counter()
                raw = infer(_FRAME_PROMPT, [frame["path"]])
                elapsed = time.perf_counter() - started
                frame_latencies.append(elapsed)
                try:
                    parsed = validate_frame_output(raw, frame)
                    validation = {"valid": True, "errors": []}
                except Exception as exc:
                    parsed = {"raw": raw}
                    validation = {"valid": False, "errors": [str(exc)]}
                frame_results.append({
                    "frame": str(frame["path"]),
                    "timestamp_seconds": frame["timestamp_seconds"],
                    "latency_seconds": round(elapsed, 3),
                    "schema_validation": validation,
                    "observation": parsed,
                })
            temporal_started = time.perf_counter()
            raw_temporal = infer(
                f"{_TEMPORAL_PROMPT}\nFrame timestamps: "
                + ", ".join(f'{f["timestamp_seconds"]:.2f}s' for f in scene["frames"])
                + ".",
                [f["path"] for f in scene["frames"]],
            )
            temporal_elapsed = time.perf_counter() - temporal_started
            temporal_latencies.append(temporal_elapsed)
            try:
                temporal = validate_temporal_output(raw_temporal)
                temporal_validation = {"valid": True, "errors": []}
            except Exception as exc:
                temporal = {"raw": raw_temporal}
                temporal_validation = {"valid": False, "errors": [str(exc)]}
            scene_results.append({
                "scene_id": scene["scene_id"],
                "elapsed_seconds": round(time.perf_counter() - scene_started, 3),
                "review_notes": scene["review_notes"],
                "expected_events": scene["expected_events"],
                "frames": frame_results,
                "temporal": {
                    "latency_seconds": round(temporal_elapsed, 3),
                    "schema_validation": temporal_validation,
                    "output": temporal,
                },
                "memory_after_scene": _mlx_memory(),
            })
        return _suite_result("qwen3-vl", model_id, load_seconds, scene_results,
                             frame_latencies, temporal_latencies)
    finally:
        del model, processor, config
        gc.collect()
        mx.clear_cache()


def _suite_result(provider: str, model_id: str, load_seconds: float,
                  scenes: list[dict[str, Any]], frame_latencies: list[float],
                  temporal_latencies: list[float]) -> dict[str, Any]:
    frame_valid = [f["schema_validation"]["valid"] for s in scenes for f in s["frames"]]
    temporal_valid = [s["temporal"]["schema_validation"]["valid"] for s in scenes]
    return {
        "provider": provider,
        "model": model_id,
        "status": "completed",
        "model_load_seconds": round(load_seconds, 3),
        "scene_count": len(scenes),
        "frame_count": len(frame_valid),
        "frame_latency": latency_summary(frame_latencies),
        "temporal_latency": latency_summary(temporal_latencies),
        "frame_schema_valid_count": sum(frame_valid),
        "frame_schema_valid_rate": round(sum(frame_valid) / len(frame_valid), 4) if frame_valid else 0,
        "temporal_schema_valid_count": sum(temporal_valid),
        "temporal_schema_valid_rate": round(sum(temporal_valid) / len(temporal_valid), 4) if temporal_valid else 0,
        "scenes": scenes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON manifest describing multiple scenes.")
    parser.add_argument("--providers", nargs="+", choices=("smolvlm2", "qwen3-vl"),
                        default=["smolvlm2", "qwen3-vl"])
    parser.add_argument("--smol-model", default=DEFAULT_SMOL_MODEL)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--output", type=Path, default=Path(".video-agent/scene-suite-benchmark.json"))
    args = parser.parse_args()
    scenes = load_manifest(args.manifest.expanduser().resolve())
    results: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "manifest": str(args.manifest.resolve()),
        "scene_count": len(scenes),
        "notes": [
            "Each candidate loads once and stays resident for the complete scene suite.",
            "Schema validity uses the production FrameObservation and SceneDynamics Pydantic models.",
            "Schema validity does not establish semantic accuracy; review raw scene outputs and annotations.",
            "Expected events and review notes are human-review aids, not automatically scored ground truth.",
            "Memory is backend-reported MLX allocation, not total macOS unified-memory pressure.",
        ],
        "runs": [],
    }
    runners = {
        "smolvlm2": lambda: run_smol_suite(scenes, args.smol_model),
        "qwen3-vl": lambda: run_qwen_suite(scenes, args.qwen_model),
    }
    for provider in args.providers:
        print(f"Running {provider} across {len(scenes)} scenes (one model load)...")
        try:
            result = runners[provider]()
        except Exception as exc:
            result = {"provider": provider, "status": "failed",
                      "error": f"{type(exc).__name__}: {exc}"}
        results["runs"].append(result)
        print(f"{provider}: {result['status']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Benchmark written to {args.output}")


if __name__ == "__main__":
    main()
