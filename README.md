# AI Video Agent

A local-first Python CLI for editing personal travel footage. The project is intentionally **not** a SaaS application: media stays on your machine, source files are never modified, and every edit will be represented by a reviewable plan before rendering.

## MVP

- Folder scanning and media metadata extraction
- Scene detection and representative-frame sampling
- Local vision-model descriptions for each detected scene
- Natural-language editing instructions (planned)
- Speech-aware pause removal that preserves quiet scenic footage (planned)
- Speech transcription and subtitles (planned)
- Automatic travel highlights (planned)
- Dry-run edit-plan review (planned)
- MP4 export through FFmpeg (planned)

## Requirements

- Python 3.11+
- FFmpeg and ffprobe installed and available on PATH
- Ollama installed and running locally for visual descriptions
- A multimodal Ollama model, default: qwen2.5vl:7b

Check the media tools and prepare the local vision model:

```bash
ffmpeg -version
ffprobe -version
ollama pull qwen2.5vl:7b
```

Keep Ollama running while using the understand command. The model runs locally; extracted frames and JSON artifacts stay in the project folder.

## Install for development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Analyze a travel-footage folder

```bash
video-agent analyze ./travel-japan
```

The command recursively discovers supported video formats, extracts metadata using ffprobe, and writes .video-agent/analysis.json. Use --output to choose another JSON destination.

## Understand video content

```bash
video-agent understand ./travel-japan
```

This command:

1. Detects scene boundaries with PySceneDetect.
2. Extracts three representative frames per scene with FFmpeg.
3. Sends those frames to your local Ollama vision model for factual descriptions.
4. Validates timestamped observations with Pydantic.
5. Writes .video-agent/understanding.json and keeps extracted JPEGs under .video-agent/frames/.

Choose a different installed vision model or sample count:

```bash
video-agent understand ./travel-japan --model qwen2.5vl:7b --samples-per-scene 2
video-agent understand ./travel-japan --output ./reports/understanding.json
```

The JSON includes each scene's start/end time, summary, representative-frame paths, visible objects, activities, and model confidence. The first version analyzes still frames independently; it does not yet infer motion across frames, identify exact locations, or understand audio. Those are planned follow-up capabilities.

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
