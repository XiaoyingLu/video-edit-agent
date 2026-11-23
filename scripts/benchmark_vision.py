"""Compare Ollama and SmolVLM2 MLX on the same representative image files.

Run this script natively on Apple Silicon for MLX acceleration:
    python scripts/benchmark_vision.py frame-1.jpg frame-2.jpg --output benchmark.json
"""

import argparse
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from video_edit_agent.atomic_io import write_json_atomic
from video_edit_agent.understanding.provider import create_vision_analyzer


def run_provider(provider: str, model: str | None, frames: list[Path]) -> dict[str, Any]:
    analyzer = create_vision_analyzer(provider, model)
    observations = []
    durations = []
    try:
        for frame in frames:
            started = time.perf_counter()
            observation = analyzer.analyze_frame(frame)
            durations.append(time.perf_counter() - started)
            observations.append({"frame": str(frame), "observation": observation})
        temporal_started = time.perf_counter()
        temporal = analyzer.analyze_scene(
            [(float(index), frame) for index, frame in enumerate(frames)]
        )
        temporal_seconds = time.perf_counter() - temporal_started
    finally:
        analyzer.unload_model()
    return {
        "provider": provider,
        "model": model,
        "frame_count": len(frames),
        "total_seconds": round(sum(durations), 3),
        "mean_seconds_per_frame": round(statistics.mean(durations), 3),
        "durations_seconds": [round(value, 3) for value in durations],
        "observations": observations,
        "temporal_analysis_seconds": round(temporal_seconds, 3),
        "temporal_analysis": temporal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "frames", nargs="+", type=Path, help="Same representative frames for each provider"
    )
    parser.add_argument("--ollama-model", default="qwen2.5vl:3b")
    parser.add_argument("--mlx-model", default="mlx-community/SmolVLM2-2.2B-Instruct-mlx")
    parser.add_argument("--skip-ollama", action="store_true")
    parser.add_argument("--skip-mlx", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("vision-benchmark.json"))
    args = parser.parse_args()

    frames = [path.expanduser().resolve() for path in args.frames]
    missing = [str(path) for path in frames if not path.is_file()]
    if missing:
        parser.error(f"Frame files do not exist: {', '.join(missing)}")

    results: dict[str, Any] = {
        "platform": platform.platform(),
        "frames": [str(path) for path in frames],
        "runs": [],
    }
    if not args.skip_ollama:
        results["runs"].append(run_provider("ollama", args.ollama_model, frames))
    if not args.skip_mlx:
        results["runs"].append(run_provider("mlx", args.mlx_model, frames))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, results)
    for run in results["runs"]:
        print(
            f"{run['provider']} ({run['model']}): "
            f"{run['mean_seconds_per_frame']:.2f}s/frame "
            f"({run['total_seconds']:.2f}s total)"
        )
    print(f"Benchmark written to {args.output}")


if __name__ == "__main__":
    main()
