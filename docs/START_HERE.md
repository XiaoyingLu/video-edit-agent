# Start here

A short checklist for returning to this repository after a break.

## 1. Set up the environment

- Install Python 3.11 or 3.12 and FFmpeg/ffprobe.
- Create and activate `.venv`.
- Install `python -m pip install -e ".[dev,transcription]"`.
- For the default vision workflow, start Ollama and pull the configured vision model.
- Copy `.env.example` to `.env`; only add hosted-provider credentials when needed.

## 2. Run a project

```bash
video-agent analyze ./my-video
video-agent understand ./my-video
video-agent transcribe ./my-video
video-agent create-plan ./my-video --prompt "Describe the video you want"
video-agent suggest-footage ./my-video
```

Review the generated artifacts under `.video-agent/`. Continue with the review,
timeline, soundtrack, and render commands only after inspecting the outputs.
See [the README](../README.md) for quickstart commands and
[the feature guide](feature-guide.md) for the full workflow.

## 3. Before opening a PR

```bash
ruff check .
ruff format --check .
pytest
docker build -t video-edit-agent:local .
```

Work on a feature branch, keep source videos and generated project artifacts out
of commits, and do not change the default vision provider without validating
quality on representative footage.
