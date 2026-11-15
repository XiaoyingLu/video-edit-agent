# AI Video Agent

A local-first Python CLI for editing personal travel footage. Media stays on your machine, source files are never modified, and every edit will be represented by a reviewable plan before rendering.

## MVP

- Folder scanning and media metadata extraction
- Scene detection and representative-frame sampling
- Local vision-model descriptions for each detected scene
- Natural-language editing, speech-aware pause removal, subtitles, highlights, edit-plan review, and MP4 export (planned)

## Requirements

- Docker Engine and Docker Compose

## Run the complete application with Docker

Both the Python CLI and Ollama/Qwen run in Docker. The CLI container mounts the current workspace at `/workspace`, so your source videos and generated `.video-agent` artifacts remain visible in the Codespace. The two containers communicate over the private Compose network; the CLI reaches Ollama at `http://ollama:11434`.

Build the CLI image and start Ollama:

```bash
docker compose build video-agent
docker compose up -d ollama
```

Download the vision model into the Ollama container (one-time download; model weights persist in a named volume):

```bash
docker compose exec ollama ollama pull qwen2.5vl:3b
docker compose exec ollama ollama list
```

For the first Codespaces test, use the lighter 3B model. On a machine with more available memory, pull `qwen2.5vl:7b` and run with `OLLAMA_MODEL=qwen2.5vl:7b`.

### Run the CLI

Put a short travel clip in a folder in the repository, for example `travel-test/`, then run:

```bash
docker compose run --rm video-agent analyze ./travel-test
docker compose run --rm video-agent understand ./travel-test --samples-per-scene 2
```

You can pass any supported CLI arguments after `video-agent`:

```bash
docker compose run --rm video-agent --help
docker compose run --rm -e OLLAMA_MODEL=qwen2.5vl:7b video-agent understand ./travel-test
```

The first command scans the folder and extracts media metadata using ffprobe. The second detects scenes, extracts two representative frames per scene, sends each frame to the local Ollama model, and writes:

```text
travel-test/
└── .video-agent/
    ├── analysis.json
    ├── understanding.json
    └── frames/
        └── <video-hash>/scene-*/frame-*.jpg
```

Open `.video-agent/understanding.json` to review timestamps, scene descriptions, visible objects, activities, and confidence values. Extracted JPEGs are useful for checking whether the sampler chose representative moments. Original video files are not modified.

### Configuration and resources

The Compose service defaults to `qwen2.5vl:3b`. Override it for a command with `-e OLLAMA_MODEL=...` as shown above. Ollama's API is also published on loopback port 11434 for optional diagnostics:

```bash
curl http://127.0.0.1:11434/api/tags
docker compose logs -f ollama
```

Qwen vision models require several gigabytes of disk and memory; inference may be slow on CPU-only Codespaces. Start with a short 30–90 second clip and one or two frames per scene. The default Compose setup makes no GPU assumption. NVIDIA acceleration requires NVIDIA Container Toolkit and an appropriate Compose GPU configuration.

### Stop services

```bash
docker compose down
```

Downloaded model weights remain in the named volume. To stop services and delete those weights too (destructive):

```bash
docker compose down -v
```

## Development

The Docker image installs the project and its runtime dependencies. To run tests and lint checks in the Codespace, install the development extra:

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
ruff format --check .
```

## Safety and privacy

- Original media is read-only from this application.
- Analysis artifacts and extracted frames are stored in the mounted workspace and ignored by Git.
- Video/audio media, generated renders, and environment files are ignored by Git.
- Do not commit personal footage, generated renders, API keys, or private transcripts.
- Vision output is descriptive evidence, not an editing instruction. The future LLM planner must produce a validated edit-plan data structure and must never execute arbitrary model-generated shell commands.

## Roadmap

1. Media scanning, scene detection, representative-frame sampling, and visual descriptions
2. Speech transcription and speech-bounded silence detection
3. Motion understanding, highlight ranking, and quality assessment
4. Validated edit-plan generation and dry-run review
5. Deterministic FFmpeg rendering and output validation
