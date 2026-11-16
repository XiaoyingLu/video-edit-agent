# AI Video Agent

A local-first Python CLI for editing personal travel footage. Media stays on your machine, source files are never modified, and every edit will be represented by a reviewable plan before rendering.

## MVP

- Folder scanning and media metadata extraction
- Scene detection and representative-frame sampling
- Local vision-model descriptions for each detected scene
- Local speech transcription with timestamped segments
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
docker compose run --rm video-agent transcribe ./travel-test --model small
```

You can pass any supported CLI arguments after `video-agent`:

```bash
docker compose run --rm video-agent --help
docker compose run --rm -e OLLAMA_MODEL=qwen2.5vl:7b video-agent understand ./travel-test
```

The `analyze` command scans the folder and extracts media metadata using ffprobe. The `understand` command detects scenes, extracts representative frames, sends each frame to the local Ollama model, and writes `.video-agent/understanding.json`. The `transcribe` command uses faster-whisper locally to recognize speech from video audio streams and writes `.video-agent/transcription.json`.

Transcription defaults to the CPU-friendly `small` model with `int8` compute. The first run downloads Whisper model weights into `.video-agent/models/`; subsequent runs reuse the cache. Use `--model tiny` for a quicker smoke test or `--device cuda --compute-type float16` on a compatible NVIDIA setup. Videos without audio are recorded in the output with an empty segment list and are not sent to Whisper.

Example output layout:

```text
travel-test/
└── .video-agent/
    ├── analysis.json
    ├── understanding.json
    ├── transcription.json
    ├── models/
    └── frames/
        └── <video-hash>/scene-*/frame-*.jpg
```

Each transcript segment contains start/end timestamps, recognized text, and optional Whisper confidence diagnostics. These timestamps will be used by the later speech-aware silence detector and subtitle generator. Original video files are not modified.

### Configuration and resources

The Compose service defaults to `qwen2.5vl:3b`. Override it for a command with `-e OLLAMA_MODEL=...`. Ollama's API is also published on loopback port 11434 for optional diagnostics:

```bash
curl http://127.0.0.1:11434/api/tags
docker compose logs -f ollama
```

Qwen vision models and Whisper weights require disk and memory; inference may be slow on CPU-only Codespaces. Start with a short 30–90 second clip, one or two frames per scene, and Whisper's `tiny` model if needed. The default Compose setup makes no GPU assumption. NVIDIA acceleration requires NVIDIA Container Toolkit and an appropriate Compose GPU configuration.

### Stop services

```bash
docker compose down
```

Downloaded Ollama weights remain in the named volume, and Whisper weights remain in the workspace cache. To stop services and delete the Ollama model volume too (destructive):

```bash
docker compose down -v
```

## Development

Install the development and transcription extras in the Codespace:

```bash
python -m pip install -e ".[dev,transcription]"
pytest
ruff check .
ruff format --check .
```

## Safety and privacy

- Original media is read-only from this application.
- Analysis artifacts, extracted frames, and local model caches are stored under `.video-agent/` and ignored by Git.
- Video/audio media, generated renders, and environment files are ignored by Git.
- Do not commit personal footage, generated renders, API keys, or private transcripts.
- Vision and transcription outputs are descriptive evidence, not editing instructions. The future LLM planner must produce a validated edit-plan data structure and must never execute arbitrary model-generated shell commands.

## Roadmap

1. Media scanning, scene detection, representative-frame sampling, and visual descriptions
2. Speech transcription and speech-bounded silence detection
3. Motion understanding, highlight ranking, and quality assessment
4. Validated edit-plan generation and dry-run review
5. Deterministic FFmpeg rendering and output validation
