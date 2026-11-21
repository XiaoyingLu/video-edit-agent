# Qwen3-VL 4B local vision provider

Qwen3-VL 4B is available as an opt-in production provider for scene-level video
understanding. It runs locally through MLX-VLM on native Apple Silicon and emits the
same `FrameObservation` and `SceneDynamics` schemas as the existing providers.

## Install

Run this in native macOS Python, not inside Docker Desktop's Linux container:

```bash
python3 -m venv .venv-mlx
source .venv-mlx/bin/activate
python -m pip install --upgrade pip
pip install -e ".[mlx,dev]"
```

MLX uses Apple's Metal backend. Confirm that it is available:

```bash
python -c "import mlx.core as mx; print(mx.metal.is_available())"
```

## Run

```bash
video-agent understand ./travel-test --provider qwen3-vl
```

The default checkpoint is
`mlx-community/Qwen3-VL-4B-Instruct-4bit`. Override it with `--model` or the
`QWEN3_VL_MODEL` environment variable:

```bash
video-agent understand ./travel-test --provider qwen3-vl \
  --model mlx-community/Qwen3-VL-4B-Instruct-4bit
```

The provider analyzes every representative frame selected by the existing scene sampler
and passes the ordered scene frames with their source timestamps to temporal analysis.
It does not reduce `--samples-per-scene`, skip temporal analysis, or change scene detection.
The existing checkpoint/resume mechanism remains active and includes provider/model identity,
so switching providers does not reuse incompatible model outputs.

## Rollout status

Qwen3-VL is opt-in. The default provider remains Ollama, and the existing `mlx` provider
continues to select SmolVLM2. Validate quality, memory pressure, and end-to-end runtime on
your own footage before choosing Qwen3-VL for a full project. The 4-bit checkpoint reduces
weight memory, but total unified-memory use also includes MLX caches, image processing,
and the rest of the application.

This integration does not claim that schema-valid output is semantically correct. Review
the generated `.video-agent/understanding.json` against the source footage.
