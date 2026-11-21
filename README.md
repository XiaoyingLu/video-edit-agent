# Video Edit Agent

A local-first AI-assisted editor for personal video projects. Analyze footage, develop a story, build a reviewable edit plan, and render only after approval. Source media is never modified.

## Quickstart

### Docker (Ollama)

Requirements: Docker Compose and a project folder containing video files.

```bash
cp .env.example .env
docker compose up -d ollama
docker compose exec ollama ollama pull qwen2.5vl:3b
docker compose exec ollama ollama pull qwen2.5:1.5b
docker compose run --rm video-agent analyze ./my-video
docker compose run --rm video-agent understand ./my-video
docker compose run --rm video-agent create-plan ./my-video --prompt "A concise, cinematic personal video"
docker compose run --rm video-agent suggest-footage ./my-video
```

For the full workflow, configuration, hosted soundtrack providers, and review steps, see [the feature guide](docs/feature-guide.md).

### Native Python

Requirements: Python 3.11 or 3.12, FFmpeg/ffprobe, and Ollama (unless using an MLX provider).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,transcription]"
ollama pull qwen2.5vl:7b
ollama pull qwen2.5:1.5b
video-agent --help
video-agent analyze ./my-video
video-agent understand ./my-video
```

For Apple Silicon MLX vision, install `.[mlx,dev]` in a native macOS environment and choose `--provider mlx` or `--provider qwen3-vl`. See [Qwen3-VL setup](docs/qwen3-vl.md).

## Command overview

- **Ingest & understand:** `analyze`, `understand`, `transcribe`, `detect-silence`, `propose-edits`, `review-edits`, `build-timeline`, `subtitles`
- **Story & planning:** `find-highlights`, `create-video`, `plan`, `create-plan`, `propose-story`, `select-story`, `suggest-footage`, `approve-storyboard`, `build-storyboard-timeline`, `review-storyboard`, `review-footage`
- **Editing & revisions:** `draft-edit`, `optimize-timeline`, `review-timeline`, `review-timeline-visual`, `apply-timeline-visual-edits`, `revise-timeline`, `review-timeline-revision`, `compare-timelines`
- **Rendering & visual review:** `render`, `init-title-overlays`, `suggest-title-sequences`, `review-title-overlays`, `render-revision-preview`, `render-draft`, `critique-draft`, `suggest-revisions`, `review-revision-suggestions`, `compare-drafts`
- **Soundtrack & music:** `suggest-soundtrack`, `search-soundtrack`, `generate-soundtrack`, `prepare-soundtrack-candidates`, `review-soundtrack`, `select-soundtrack`, `analyze-selected-soundtrack`, `analyze-music`, `suggest-music-sync`, `review-music-timeline`, `review-music-sync`
- **Project maintenance:** `validate-artifacts`, `clean`, `report-revision-cycle`

Run `video-agent --help` for per-command options.

## Pipeline

```mermaid
flowchart LR
  A["Source videos"] --> B["Analyze & understand"]
  B --> C["Story proposal & plan"]
  C --> D["Suggest footage"]
  D --> E["Review & approve"]
  E --> F["Timeline & soundtrack"]
  F --> G["Render & inspect"]
```

## Project layout

- `src/travel_video_agent/` — Python import namespace (retained for compatibility).
- `docs/` — architecture, model setup, feature guides, maintainer notes, and release instructions.
- `.video-agent/` — project-local analysis, plans, review decisions, and generated outputs.

The distribution and CLI are named `video-edit-agent` / `video-agent`; the Python import namespace remains `travel_video_agent` to avoid breaking existing integrations.
