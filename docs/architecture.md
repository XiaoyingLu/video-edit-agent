# Architecture

The application is a local-first CLI. Media processing is deterministic where possible; visual interpretation is delegated to an optional local multimodal model.

## Current media-analysis flow

```text
CLI (Typer)
  -> folder discovery + ffprobe metadata
  -> PySceneDetect scene boundaries
  -> FFmpeg representative-frame extraction
  -> local Ollama vision model
  -> Pydantic validation
  -> .video-agent/understanding.json + cached frame JPEGs
```

## Video understanding boundaries

- PySceneDetect proposes temporal scene boundaries; it does not describe scene meaning.
- FFmpeg extracts a small number of representative still frames per scene.
- Ollama describes only visible evidence in each still frame and returns structured JSON.
- Each observation retains its timestamp, frame path, and confidence.
- The first iteration does not infer motion between frames or identify exact geographic locations.

The vision model is accessed through an adapter so other local or hosted providers can be added without changing the pipeline. The default Ollama model is qwen2.5vl:7b.

## Speech transcription

- faster-whisper runs locally and returns timestamped speech segments with language and confidence metadata.
- Long gaps before, between, and after speech are recorded as candidate pause intervals.
- A gap is not proof of dead air: ambient sound, music, and quiet scenic footage may continue through it.
- Pause metadata never triggers a cut. The user must review a future typed edit plan before rendering.

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

## Travel-footage safety rule

Silence is not equivalent to dead air. Speech gaps are descriptive metadata only; quiet scenic footage, ambient sound, and music are retained unless the user explicitly requests otherwise.
