# Architecture

The application is a local-first CLI. The first slice is deliberately deterministic and does not require an LLM.

## Current flow

```text
CLI (Typer)
  -> folder discovery
  -> ffprobe metadata extraction
  -> Pydantic validation
  -> local analysis.json
```

## Planned editing workflow

```text
Input folder + natural-language instruction
  -> media analysis (scenes, sampled frames, speech)
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
