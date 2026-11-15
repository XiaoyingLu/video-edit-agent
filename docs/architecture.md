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
- The first iteration does not infer motion between frames, transcribe audio, or identify exact geographic locations.

The vision model is accessed through an adapter so other local or hosted providers can be added without changing the pipeline. The default Ollama model is qwen2.5vl:7b.

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

Silence is not equivalent to dead air. Silence removal is restricted to detected speech regions; quiet scenic footage is retained unless the user explicitly requests otherwise.
