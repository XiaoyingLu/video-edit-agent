# AI Video Agent

A local-first Python CLI for editing personal travel footage. Media stays on your machine, source files are never modified, and every edit will be represented by a reviewable plan before rendering.

## MVP

- Folder scanning and media metadata extraction
- Scene detection and representative-frame sampling
- Local vision-model descriptions for each detected scene
- Local speech transcription with timestamped segments and long-gap metadata
- Human review of proposed cuts and validated retained-segment timeline
- Timeline-aligned SRT subtitles and optional burned-in subtitles
- Explainable visual highlight candidate detection and ranking
- Natural-language editing intent parsing into a validated, reviewable plan

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
docker compose run --rm video-agent detect-silence ./travel-test --noise-db -35 --min-duration 1.5
docker compose run --rm video-agent propose-edits ./travel-test
docker compose run --rm -it video-agent review-edits ./travel-test
docker compose run --rm video-agent build-timeline ./travel-test
docker compose run --rm video-agent find-highlights ./travel-test --samples-per-scene 3
docker compose run --rm video-agent plan ./travel-test --prompt "Create a 60-second cinematic travel video"
docker compose run --rm video-agent draft-edit ./travel-test
docker compose run --rm video-agent optimize-timeline ./travel-test
docker compose run --rm -it video-agent review-timeline ./travel-test
docker compose run --rm video-agent subtitles ./travel-test
docker compose run --rm video-agent render ./travel-test --burn-subtitles --subtitle-font-size 28 --subtitle-position bottom
docker compose run --rm video-agent render-draft ./travel-test
```

You can pass any supported CLI arguments after `video-agent`:

```bash
docker compose run --rm video-agent --help
docker compose run --rm -e OLLAMA_MODEL=qwen2.5vl:7b video-agent understand ./travel-test
```

The `render-draft` command renders the approved `.video-agent/optimized-timeline.json` to `.video-agent/rendered-draft.mp4`. It refuses to run until every proposed shot has an explicit decision and at least one shot is approved. It preserves the reviewed shot order, applies crossfade/dissolve video transitions with matching audio crossfades, uses hard cuts where a transition is not approved or its neighboring shot was rejected, synthesizes silence for clips without audio, probes the finished MP4, and writes it atomically without modifying source footage. It has a separate output and does not change the legacy `render` command. The `optimize-timeline` command turns the selected clips into an ordered shot sequence, proposes style-aware transitions, and flags shots that fall outside the requested pacing range. It writes `.video-agent/optimized-timeline.json`, a review artifact deliberately separate from the renderer's `timeline.json`. Use `review-timeline` to approve or reject each shot interactively, or pass repeated `--approve CLIP_ID` and `--reject CLIP_ID` options. The sequence is approved only after every proposed clip has an explicit decision and at least one clip is approved. Source intervals are preserved exactly; pacing notes are recommendations rather than automatic trims. This stage does not render or alter source media. The `plan` command sends a natural-language editing request to the local Ollama model and validates its structured response against a strict Pydantic schema, writing `.video-agent/editing-plan.json`. The `draft-edit` command combines that plan with ranked visual highlights, optional scene descriptions, and optional transcript excerpts to produce a duration-bounded, reviewable shot selection in `.video-agent/draft-selection.json`. Selections are recommendations only: the command does not change the approved timeline or render a video. The `find-highlights` command detects scenes, measures sampled-frame motion, sharpness, exposure, and saturation, then writes ranked, explainable candidates to `.video-agent/highlights.json`. The score is a deterministic visual heuristic, not an aesthetic judgment; every candidate remains review-only, and low-motion scenic footage is explicitly preserved as a possible highlight. The `analyze` command scans the folder and extracts media metadata using ffprobe. The `understand` command detects scenes, extracts representative frames, sends each frame to the local Ollama model, and writes `.video-agent/understanding.json`. The `transcribe` command uses faster-whisper locally to recognize speech from video audio streams and writes `.video-agent/transcription.json`. The `detect-silence` command runs FFmpeg’s `silencedetect` filter and writes `.video-agent/silence-analysis.json`, correlating quiet intervals with transcript timestamps when available. The `propose-edits` command combines those review candidates with optional scene descriptions and writes a human-reviewable `.video-agent/edit-proposal.json`. Use `review-edits` to approve/reject each item interactively, or pass repeated `--approve ITEM_ID` and `--reject ITEM_ID` options for scripted review. Skipped items remain pending. Once every item has an explicit decision, `build-timeline` writes `.video-agent/timeline.json` containing the ordered source intervals to retain. This validated timeline is an edit-decision artifact only. After every cut has been reviewed, `render` uses FFmpeg to create `.video-agent/rendered.mp4` as an H.264/AAC MP4. The renderer checks the timeline and source files, keeps all source media untouched, refuses to overwrite an existing output unless `--overwrite` is supplied, and probes the finished MP4 before reporting success. Clips with no audio receive generated silence so all segments can be concatenated consistently. The subtitles command reads the Whisper transcription and validated timeline, remaps speech captions through removed intervals and concatenated clips, and writes .video-agent/subtitles.srt. Render with --burn-subtitles to burn the SRT into the final output after concatenation; use --subtitle-font-size (12–72) and --subtitle-position (bottom, center, top) to adjust its appearance, or pass --subtitle-file for a custom SRT.

Transcription defaults to the CPU-friendly `small` model with `int8` compute. The first run downloads Whisper model weights into `.video-agent/models/`; subsequent runs reuse the cache. Use `--model tiny` for a quicker smoke test or `--device cuda --compute-type float16` on a compatible NVIDIA setup. Videos without audio are recorded in the output with an empty segment list and are not sent to Whisper.

Example output layout:

```text
travel-test/
└── .video-agent/
    ├── analysis.json
    ├── understanding.json
    ├── transcription.json
    ├── silence-analysis.json
    ├── edit-proposal.json
    ├── timeline.json
    ├── highlights.json
    ├── draft-selection.json
    ├── optimized-timeline.json
    ├── subtitles.srt
    ├── rendered.mp4
    ├── models/
    └── frames/
        └── <video-hash>/scene-*/frame-*.jpg
```

Each transcript segment contains start/end timestamps, recognized text, and optional Whisper confidence diagnostics. The artifact also reports long gaps before, between, and after speech segments. These gaps are descriptive only—not automatic cut suggestions. Silence analysis reports low-volume intervals and emits review candidates only when a transcript is available and no speech overlaps the interval (with a configurable safety margin). Quiet intervals may still contain scenic ambience, music, wind, or other intentional sound: candidates are never automatic cuts and always require human review. Original video files are not modified.

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

The Docker image includes the development tools. Run tests and lint without starting Ollama:

```bash
docker compose build video-agent
docker compose run --rm --no-deps --entrypoint python video-agent -m pytest
docker compose run --rm --no-deps --entrypoint ruff video-agent check .
docker compose run --rm --no-deps --entrypoint ruff video-agent format --check .
```

For a non-Docker development environment, install the development and transcription extras:

```bash
python -m pip install -e ".[dev,transcription]"
pytest
ruff check .
ruff format --check .
```

## Safety and privacy

- Original media is read-only from this application. Rendering writes to a separate output file and never uses a source path as its destination.
- Legacy rendering is gated on a validated timeline with no pending review items. Draft rendering additionally requires every proposed shot to be explicitly approved or rejected, with at least one approved shot. Existing MP4 outputs are not replaced unless `--overwrite` is explicitly supplied.
- Analysis artifacts, extracted frames, and local model caches are stored under `.video-agent/` and ignored by Git.
- Video/audio media, generated renders, and environment files are ignored by Git.
- Do not commit personal footage, generated renders, API keys, or private transcripts.
- Vision and transcription outputs are descriptive evidence, not editing instructions. The future LLM planner must produce a validated edit-plan data structure and must never execute arbitrary model-generated shell commands.

## Roadmap

1. Media scanning, scene detection, representative-frame sampling, and visual descriptions
2. Speech transcription and speech-bounded silence detection
3. Motion understanding, explainable highlight ranking, and quality assessment (initial visual-signal MVP implemented)
4. Validated edit-plan generation and dry-run review
5. Deterministic FFmpeg rendering and output validation (implemented)
6. Timeline-aligned SRT generation and optional burned-in subtitles (implemented)
7. Natural-language request parsing into a validated editing plan (implemented)
8. Context-aware clip selection (implemented)
9. Ordered shot timeline, transition proposals, pacing guidance, and human review (implemented)
10. Render approved shot timelines with FFmpeg transitions and output verification (implemented; Docker validation pending)
