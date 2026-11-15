# AI Video Agent

A local-first Python CLI for editing personal travel footage. Media stays on your machine, source files are never modified, and every edit will be represented by a reviewable plan before rendering.

## MVP

- Folder scanning and media metadata extraction
- Scene detection and representative-frame sampling
- Local vision-model descriptions for each detected scene
- Natural-language editing, speech-aware pause removal, subtitles, highlights, edit-plan review, and MP4 export (planned)

## Requirements

- Python 3.11+
- FFmpeg and ffprobe on PATH
- Docker Engine and Docker Compose

## Run the local vision model with Docker

The recommended development setup runs **Ollama/Qwen in Docker** and runs the Python CLI in your Codespace or local Python environment. The Compose service publishes the Ollama API only on loopback port 11434 and persists downloaded models in a named Docker volume.

Start Ollama:

```bash
docker compose up -d
```

Download the vision model into the running container (this is a one-time download; the Docker volume persists it):

```bash
docker compose exec ollama ollama pull qwen2.5vl:3b
```

Confirm that Ollama responds and the model is installed:

```bash
curl http://127.0.0.1:11434/api/tags
docker compose exec ollama ollama list
```

Optional: set the endpoint/model explicitly in your shell:

```bash
export OLLAMA_BASE_URL=http://127.0.0.1:11434
export OLLAMA_MODEL=qwen2.5vl:3b
```

The README uses the 3B variant as a lighter first test in a Codespace. On a machine with more available memory, you can pull `qwen2.5vl:7b` and set `OLLAMA_MODEL=qwen2.5vl:7b`. The CLI defaults to `qwen2.5vl:7b` when `OLLAMA_MODEL` is not set. If the Python application itself is later run inside a separate Compose container on the same Compose network, set `OLLAMA_BASE_URL=http://ollama:11434` instead of localhost.

### Resource notes

Qwen2.5-VL 7B is a substantial model. First download can take several gigabytes, and inference may be slow on CPU-only Codespaces. Ensure the Codespace has sufficient memory and disk before pulling it. If resource usage is too high, use a smaller compatible Qwen2.5-VL model tag supported by Ollama and set `OLLAMA_MODEL` (or pass `--model`) accordingly. For NVIDIA acceleration, Docker must have NVIDIA Container Toolkit configured; the default Compose file intentionally makes no GPU assumption.

## Install the Python CLI

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Test with a real travel clip

Put one or more supported videos (`.mp4`, `.mov`, `.mkv`, `.avi`, `.m4v`, `.webm`) in a folder accessible to the Python environment. For the first test, use a short 30–90 second clip to validate the end-to-end path before processing a long trip.

```bash
video-agent analyze ./travel-test
video-agent understand ./travel-test --samples-per-scene 2
```

The command detects scene boundaries, extracts two representative frames per scene, sends each frame to the local Ollama model, and writes:

```text
travel-test/
└── .video-agent/
    ├── analysis.json
    ├── understanding.json
    └── frames/
        └── <video-hash>/scene-*/frame-*.jpg
```

Open `.video-agent/understanding.json` to review timestamps, scene descriptions, visible objects, activities, and confidence values. Extracted JPEGs are useful for checking whether the sampler chose representative moments. Original video files are not modified.

## Useful commands

```bash
# Follow model server logs
docker compose logs -f ollama

# Stop the model container; downloaded weights remain in the named volume
docker compose down

# Stop and delete the downloaded model volume (destructive)
docker compose down -v

# Choose a model and fewer frames per scene
video-agent understand ./travel-test --model qwen2.5vl:3b --samples-per-scene 1
```

## Development

```bash
pytest
ruff check .
ruff format --check .
```

## Safety and privacy

- Original media is read-only from this application.
- Analysis artifacts and extracted frames are stored locally and ignored by Git.
- Video/audio media, generated renders, and environment files are ignored by Git.
- Do not commit personal footage, generated renders, API keys, or private transcripts.
- Vision output is descriptive evidence, not an editing instruction. The future LLM planner must produce a validated edit-plan data structure and must never execute arbitrary model-generated shell commands.

## Roadmap

1. Media scanning, scene detection, representative-frame sampling, and visual descriptions
2. Speech transcription and speech-bounded silence detection
3. Motion understanding, highlight ranking, and quality assessment
4. Validated edit-plan generation and dry-run review
5. Deterministic FFmpeg rendering and output validation
