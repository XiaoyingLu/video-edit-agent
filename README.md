# AI Video Agent

A local-first Python CLI for editing personal travel footage. The project is intentionally **not** a SaaS application: media stays on your machine, source files are never modified, and every edit will be represented by a reviewable plan before rendering.

## MVP

- Natural-language editing instructions (planned)
- Speech-aware pause removal that preserves quiet scenic footage (planned)
- Speech transcription and subtitles (planned)
- Automatic travel highlights (planned)
- Dry-run edit-plan review (planned)
- MP4 export through FFmpeg (planned)
- First implemented slice: inspect a folder of videos and save metadata locally

## Requirements

- Python 3.11+
- FFmpeg and ffprobe installed and available on PATH

Check the media tools:

```bash
ffmpeg -version
ffprobe -version
```

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

The command recursively discovers supported video formats, extracts metadata using ffprobe, and writes:

```text
travel-japan/
└── .video-agent/
    └── analysis.json
```

Use `--output` to choose another JSON destination:

```bash
video-agent analyze ./travel-japan --output ./reports/travel-analysis.json
```

The analysis file contains relative file paths, duration, dimensions, frame rate, video/audio codecs, and stream-presence flags. It does not copy or alter the source videos.

## Development

```bash
pytest
ruff check .
ruff format --check .
```

## Safety and privacy

- Original media is read-only from this application.
- Analysis artifacts are stored locally and ignored by Git.
- Video/audio media, generated exports, and environment files are ignored by Git.
- Do not commit personal footage, generated renders, API keys, or private transcripts.
- The LLM planner must produce a validated edit-plan data structure; it must never execute arbitrary model-generated shell commands.

## Roadmap

1. Media scanning and ffprobe metadata extraction
2. Thumbnail/contact-sheet generation and scene detection
3. Speech transcription and speech-bounded silence detection
4. Highlight ranking and validated edit-plan generation
5. Dry-run review and deterministic FFmpeg rendering
