# AI Video Agent

A local-first Python CLI for editing personal travel footage. Media stays in the mounted workspace, source files are never modified, and every future edit will be represented by a reviewable plan before rendering.

## Current capabilities

- Recursively scan video folders and extract metadata with ffprobe
- Detect scenes and sample representative frames
- Describe sampled frames with a local Ollama vision model
- Transcribe speech locally with faster-whisper
- Produce timestamped speech segments and identify long gaps between speech

Natural-language editing, automatic highlight ranking, subtitle export, edit-plan review, and MP4 rendering are planned.

## Run with Docker

Both the Python CLI and Ollama run in Docker. The CLI container mounts the workspace at `/workspace`; source videos and generated `.video-agent` artifacts remain accessible in the Codespace.

Build the image and start Ollama:

```bash
docker compose build video-agent
docker compose up -d ollama
```

Pull the local vision model once (weights persist in the named volume):

```bash
docker compose exec ollama ollama pull qwen2.5vl:3b
docker compose exec ollama ollama list
```

Put a short clip in `travel-test/`, then run:

```bash
docker compose run --rm video-agent analyze ./travel-test
docker compose run --rm video-agent understand ./travel-test --samples-per-scene 2
docker compose run --rm video-agent transcribe ./travel-test
```

The first two commands create `.video-agent/analysis.json` and `.video-agent/understanding.json`. Transcription creates `.video-agent/transcription.json`.

### Transcription options

```bash
# Use the multilingual tiny model (default), CPU int8
docker compose run --rm video-agent transcribe ./travel-test

# Specify language to improve decoding when it is known
docker compose run --rm video-agent transcribe ./travel-test --language en

# Use a larger model for potentially better accuracy (more memory and disk)
docker compose run --rm video-agent transcribe ./travel-test --model small

# Adjust what counts as a long pause
docker compose run --rm video-agent transcribe ./travel-test --min-pause-duration 1.5
```

The first transcription run downloads the selected faster-whisper model into the Hugging Face model cache in the container. To retain that cache between one-off CLI containers, mount a named volume at `/root/.cache/huggingface` in the `video-agent` service. Without that additional volume, the model may be downloaded again on a later run.

The transcription artifact contains per-video language metadata, speech segments, and gaps between speech. A gap is descriptive metadata only: **it is not a cut instruction**. Quiet scenic footage, ambient sound, music, and pauses must be preserved unless the user explicitly asks to remove them and a reviewed edit plan says so.

## Configuration and resources

The Compose vision model defaults to `qwen2.5vl:3b`; override it with `-e OLLAMA_MODEL=qwen2.5vl:7b` if your machine has sufficient resources. CPU transcription defaults to the `tiny` model with int8 compute. Larger Whisper models use more memory and run time. For a first test, use a 30–90 second clip.

```bash
docker compose run --rm video-agent --help
docker compose logs -f ollama
docker compose down
```

Downloaded Ollama weights persist in `ollama_data`. `docker compose down -v` deletes that volume.

## Development

```bash
python -m pip install -e ".[dev,audio]"
pytest
ruff check .
ruff format --check .
```

## Safety and privacy

- Source media is read-only; FFmpeg writes temporary extracted audio and derived artifacts only.
- Audio is processed locally by faster-whisper; it is not sent to a hosted transcription API.
- Transient WAV files are deleted after transcription.
- Analysis artifacts, extracted frames, and transcripts are stored under `.video-agent` and ignored by Git.
- Never commit personal footage, transcripts, generated renders, or credentials.
- Vision and transcription outputs are evidence, not editing instructions. Future model planning must produce validated typed plans; arbitrary model-generated shell commands are never executed.

## Roadmap

1. Media scanning, visual scene understanding, and local speech transcription
2. Motion understanding, highlight ranking, and quality assessment
3. Validated edit-plan generation and dry-run timeline review
4. Deterministic FFmpeg rendering, subtitles, and output validation
