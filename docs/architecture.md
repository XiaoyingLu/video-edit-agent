# Architecture

The application is a local-first CLI. Media processing is deterministic where possible; local models provide visual descriptions and speech transcription. Original media is never modified.

## Current analysis flows

Visual understanding:

```text
CLI (Typer)
  -> folder discovery + ffprobe metadata
  -> PySceneDetect scene boundaries
  -> FFmpeg representative-frame extraction
  -> local Ollama vision model
  -> Pydantic validation
  -> .video-agent/understanding.json + cached frame JPEGs
```

Audio understanding:

```text
CLI (Typer)
  -> folder discovery + ffprobe metadata
  -> FFmpeg extracts temporary mono 16 kHz WAV
  -> local faster-whisper (CPU/int8 by default)
  -> timestamped speech segments + language metadata
  -> derive gaps between speech segments
  -> Pydantic validation
  -> .video-agent/transcription.json
  -> delete temporary WAV
```

## Speech timeline semantics

- Speech segments are timestamped model output, not ground truth.
- Pause intervals are gaps before, between, and after detected speech.
- A pause is **not** automatically classified as dead air or safe to remove.
- Quiet scenic footage, ambient sound, music, and intentional pauses must be preserved by default.
- The future edit planner may propose edits only inside explicitly permitted regions; the user reviews the typed plan before rendering.

## Model boundaries

- PySceneDetect proposes scene boundaries; it does not describe scene meaning.
- Ollama describes visible evidence in sampled still frames. The first iteration does not infer motion or exact geographic locations.
- faster-whisper runs locally and returns speech segments and language metadata. Its timestamps and confidence are estimates.
- Model outputs are validated with Pydantic before becoming project artifacts.

## Planned editing workflow

```text
Input folder + natural-language instruction
  -> media understanding (visual scenes, speech, motion, quality)
  -> highlight ranking
  -> LLM planner (structured edit-plan output)
  -> deterministic plan validation
  -> dry-run timeline review
  -> FFmpeg renderer
  -> output validation and report
```

The LLM must not generate shell commands for direct execution. It may only propose a typed edit plan. All file paths, timestamps, codecs, and output settings are validated before deterministic tools execute.
