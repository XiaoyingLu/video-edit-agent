"""Compare local video-understanding model candidates on identical scene frames.

Run natively on Apple Silicon:
    python scripts/benchmark_model_candidates.py frame-1.jpg frame-2.jpg frame-3.jpg

The benchmark keeps model adapters isolated from production provider selection. It
records raw outputs for human review; it does not declare semantic equivalence.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

from travel_video_agent.understanding.smolvlm2_analyzer import (
    _FRAME_PROMPT,
    _TEMPORAL_PROMPT,
    SmolVLM2MLXAnalyzer,
)

DEFAULT_SMOL_MODEL = "mlx-community/SmolVLM2-2.2B-Instruct-mlx"
DEFAULT_QWEN_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
DEFAULT_VIDEOCHAT_MODEL = "MCG-NJU/VideoChat3-4B"


def _as_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    raise RuntimeError(f"Unsupported generation result: {type(result).__name__}")


def _is_json_object(value: Any) -> bool:
    if isinstance(value, dict):
        return True
    if not isinstance(value, str):
        return False
    candidate = value.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1].removesuffix("```").strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        return False
    try:
        return isinstance(json.loads(candidate[start : end + 1]), dict)
    except json.JSONDecodeError:
        return False


def _reset_mlx_peak_memory() -> None:
    try:
        import mlx.core as mx

        mx.reset_peak_memory()
    except (ImportError, AttributeError):
        pass


def _mlx_memory_mb() -> dict[str, float]:
    try:
        import mlx.core as mx

        return {
            "active_mb": round(mx.get_active_memory() / 1024**2, 1),
            "cache_mb": round(mx.get_cache_memory() / 1024**2, 1),
            "peak_mb": round(mx.get_peak_memory() / 1024**2, 1),
        }
    except (ImportError, AttributeError):
        return {}


def _torch_memory_mb(torch: Any, device: str) -> dict[str, float]:
    if device != "mps":
        return {}
    try:
        return {
            "current_allocated_mb": round(torch.mps.current_allocated_memory() / 1024**2, 1),
            "driver_allocated_mb": round(torch.mps.driver_allocated_memory() / 1024**2, 1),
        }
    except (AttributeError, RuntimeError):
        return {}


def run_mlx_candidate(
    provider: str, model_id: str, frames: list[Path]
) -> dict[str, Any]:
    """Run a MLX-VLM model using the same single-frame and ordered multi-image prompts."""
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    load_started = time.perf_counter()
    model, processor = load(model_id)
    config = load_config(model_id)
    load_seconds = time.perf_counter() - load_started
    _reset_mlx_peak_memory()

    def infer(prompt: str, image_paths: list[Path]) -> str:
        formatted = apply_chat_template(
            processor, config, prompt, num_images=len(image_paths)
        )
        return _as_text(
            generate(
                model,
                processor,
                formatted,
                [str(path) for path in image_paths],
                max_tokens=1024,
                temp=0.0,
            )
        )

    try:
        frame_results = []
        frame_durations = []
        for frame in frames:
            started = time.perf_counter()
            output = infer(_FRAME_PROMPT, [frame])
            frame_durations.append(time.perf_counter() - started)
            frame_results.append(
                {"frame": str(frame), "output": output, "json_valid": _is_json_object(output)}
            )

        temporal_started = time.perf_counter()
        temporal_prompt = (
            f"{_TEMPORAL_PROMPT}\nFrame timestamps: "
            + ", ".join(f"{index:.2f}s" for index in range(len(frames)))
            + "."
        )
        temporal_output = infer(temporal_prompt, frames)
        temporal_seconds = time.perf_counter() - temporal_started
        return {
            "provider": provider,
            "model": model_id,
            "device": "mlx-metal",
            "model_load_seconds": round(load_seconds, 3),
            "frame_durations_seconds": [round(value, 3) for value in frame_durations],
            "frame_total_seconds": round(sum(frame_durations), 3),
            "frame_mean_seconds": round(statistics.mean(frame_durations), 3),
            "frames": frame_results,
            "temporal_analysis_seconds": round(temporal_seconds, 3),
            "temporal_output": temporal_output,
            "temporal_json_valid": _is_json_object(temporal_output),
            "memory": _mlx_memory_mb(),
        }
    finally:
        del model, processor, config
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, AttributeError):
            pass


def run_videochat3_candidate(
    model_id: str, frames: list[Path], device: str
) -> dict[str, Any]:
    """Run VideoChat3 through its published Transformers/processor interface."""
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from qwen_vl_utils import process_vision_info

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is not available on this machine.")

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "mps" else "auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    load_seconds = time.perf_counter() - load_started

    def infer(prompt: str, image_paths: list[Path]) -> str:
        content = [
            {"type": "image", "image": str(path)} for path in image_paths
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=14,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        video_metadata = None
        if videos is not None:
            videos, video_metadata = zip(*videos)
            videos, video_metadata = list(videos), list(video_metadata)
        inputs = processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            do_resize=False,
            return_tensors="pt",
            **(video_kwargs or {}),
        ).to(device)
        if hasattr(model, "dtype"):
            inputs = inputs.to(model.dtype)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=1024, do_sample=False
            )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated)
        ]
        return processor.tokenizer.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    try:
        frame_results = []
        frame_durations = []
        for frame in frames:
            started = time.perf_counter()
            output = infer(_FRAME_PROMPT, [frame])
            frame_durations.append(time.perf_counter() - started)
            frame_results.append(
                {"frame": str(frame), "output": output, "json_valid": _is_json_object(output)}
            )

        temporal_started = time.perf_counter()
        temporal_prompt = (
            f"{_TEMPORAL_PROMPT}\nFrame timestamps: "
            + ", ".join(f"{index:.2f}s" for index in range(len(frames)))
            + "."
        )
        temporal_output = infer(temporal_prompt, frames)
        temporal_seconds = time.perf_counter() - temporal_started
        return {
            "provider": "videochat3",
            "model": model_id,
            "device": device,
            "model_load_seconds": round(load_seconds, 3),
            "frame_durations_seconds": [round(value, 3) for value in frame_durations],
            "frame_total_seconds": round(sum(frame_durations), 3),
            "frame_mean_seconds": round(statistics.mean(frame_durations), 3),
            "frames": frame_results,
            "temporal_analysis_seconds": round(temporal_seconds, 3),
            "temporal_output": temporal_output,
            "temporal_json_valid": _is_json_object(temporal_output),
            "memory": _torch_memory_mb(torch, device),
        }
    finally:
        del model, processor
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()


def run_smol_baseline(model_id: str, frames: list[Path]) -> dict[str, Any]:
    """Run the merged production SmolVLM2 analyzer as the baseline."""
    analyzer = SmolVLM2MLXAnalyzer(model=model_id)
    frame_results = []
    frame_durations = []
    try:
        load_started = time.perf_counter()
        analyzer._load()
        load_seconds = time.perf_counter() - load_started
        _reset_mlx_peak_memory()
        for frame in frames:
            started = time.perf_counter()
            observation = analyzer.analyze_frame(frame)
            frame_durations.append(time.perf_counter() - started)
            frame_results.append(
                {"frame": str(frame), "observation": observation, "json_valid": True}
            )
        temporal_started = time.perf_counter()
        temporal = analyzer.analyze_scene(
            [(float(index), frame) for index, frame in enumerate(frames)]
        )
        temporal_seconds = time.perf_counter() - temporal_started
        return {
            "provider": "smolvlm2",
            "model": model_id,
            "device": "mlx-metal",
            "model_load_seconds": round(load_seconds, 3),
            "frame_durations_seconds": [round(value, 3) for value in frame_durations],
            "frame_total_seconds": round(sum(frame_durations), 3),
            "frame_mean_seconds": round(statistics.mean(frame_durations), 3),
            "frames": frame_results,
            "temporal_analysis_seconds": round(temporal_seconds, 3),
            "temporal_output": temporal,
            "temporal_json_valid": True,
            "memory": _mlx_memory_mb(),
        }
    finally:
        analyzer.unload_model()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", nargs="+", type=Path)
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=("smolvlm2", "qwen3-vl", "videochat3"),
        default=["smolvlm2", "qwen3-vl"],
        help="Candidates to run. VideoChat3 is opt-in because its weights are about 9 GB.",
    )
    parser.add_argument("--smol-model", default=DEFAULT_SMOL_MODEL)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--videochat3-model", default=DEFAULT_VIDEOCHAT_MODEL)
    parser.add_argument("--videochat3-device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--output", type=Path, default=Path("model-candidate-benchmark.json"))
    args = parser.parse_args()

    frames = [path.expanduser().resolve() for path in args.frames]
    missing = [str(path) for path in frames if not path.is_file()]
    if missing:
        parser.error(f"Frame files do not exist: {', '.join(missing)}")
    if len(frames) < 2:
        parser.error("Provide at least two ordered frames for temporal analysis.")

    results: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "frames": [str(path) for path in frames],
        "notes": [
            "Use the same ordered frames for every candidate.",
            "Frame timestamps in this benchmark are ordinal placeholders, not source-video timestamps.",
            "Model load time includes first-time checkpoint loading but excludes downloads already cached.",
            "Memory values are backend-reported allocations, not total system memory.",
            "Raw outputs require human quality review; JSON validity is not schema validity.",
        ],
        "runs": [],
    }

    runners = {
        "smolvlm2": lambda: run_smol_baseline(args.smol_model, frames),
        "qwen3-vl": lambda: run_mlx_candidate("qwen3-vl", args.qwen_model, frames),
        "videochat3": lambda: run_videochat3_candidate(
            args.videochat3_model, frames, args.videochat3_device
        ),
    }
    for provider in args.providers:
        print(f"Running {provider}...")
        try:
            run = runners[provider]()
        except Exception as exc:  # A failed candidate should not discard other results.
            run = {
                "provider": provider,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        run.setdefault("status", "completed")
        results["runs"].append(run)
        print(f"{provider}: {run['status']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Benchmark written to {args.output}")


if __name__ == "__main__":
    main()
