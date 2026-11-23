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
- FFmpeg’s `silencedetect` filter reports low-volume intervals using configurable dB and duration thresholds.
- Quiet intervals overlapping transcript speech (including a configurable safety margin) are excluded from review candidates.
- Without a transcript artifact, only audio-level observations are emitted; no candidates are proposed.
- Every candidate remains review-only because quiet scenic footage, ambience, and music can be intentional. No source media is modified.

## Reviewable edit proposals

- `propose-edits` reads the validated silence-analysis artifact and optionally joins visual scene summaries by the source-video hash embedded in scene IDs.
- Each proposal item retains source-relative timestamps, a stable review ID, optional scene context, and an explicit pending decision.
- If any overlapping scene or frame observation is classified as `scenic_breath` or `establishing`, the proposal uses the typed `suggested_action="preserve"` hint and records the scene role. This is deliberately conservative: the item remains pending and review-required, and only an explicit human approval can turn it into a cut.
- Other quiet intervals may be labeled `suggested_action="consider_removing"`; this is still a suggestion, not an edit decision.
- A quiet interval is only a review candidate. The proposal never marks it approved, never changes source media, and never invokes FFmpeg rendering.
- The proposal is serialized to `.video-agent/edit-proposal.json`. The `review-edits` command records explicit approvals/rejections; skipped items remain pending. Timeline generation refuses to proceed while any item is pending.
- `build-timeline` subtracts approved cut intervals from each source duration and writes `.video-agent/timeline.json` with ordered retained source intervals. Rejected candidates remain in the retained footage.
- The timeline schema validates non-negative, in-bounds, ordered, non-overlapping segments and retained-duration totals. Approved cuts are checked for overlap and source bounds.
- The timeline is a validated edit-decision artifact. `render` consumes only this artifact; it does not accept LLM output or arbitrary commands. Before rendering, the renderer revalidates every source path, source duration, segment boundary, and output destination. Source paths must resolve inside the project folder, and the destination cannot resolve to a source file.
- FFmpeg trims each retained interval, resets timestamps, normalizes video dimensions/frame rate/pixel format and audio sample rate/channel layout, then concatenates the segments in timeline order. Sources without audio receive generated silence to keep the output stream layout consistent.
- The renderer encodes H.264 video and AAC audio into a temporary MP4, probes the result for valid streams and expected duration, then atomically moves it to the destination. It refuses to overwrite an existing output unless `--overwrite` is explicitly passed. Failed renders clean up the temporary output; source media is never modified.

## Timeline-aligned subtitles

- The subtitles command reads the validated timeline and timestamped Whisper transcription; it does not run recognition again.
- Source-time speech cues are intersected with retained timeline intervals and shifted into the final concatenated output timebase. Speech entirely inside removed intervals is omitted. A cue crossing a cut is split across the retained fragments.
- The generated .video-agent/subtitles.srt is a sidecar artifact and does not modify source media.
- render --burn-subtitles applies FFmpeg's libass-backed subtitles filter after segment concatenation, ensuring cue timestamps refer to the final output rather than individual source clips. Font size and vertical placement are configurable.
- The Docker image installs DejaVu Sans for predictable default subtitle typography. Rendering without subtitles remains unchanged.

## Planned editing workflow

```text
Input folder + natural-language instruction
  -> media understanding (visual scenes, speech, motion, quality)
  -> highlight ranking
  -> LLM planner (structured edit-plan output)
  -> deterministic plan validation
  -> dry-run timeline review
  -> validated retained-segment timeline
  -> timeline-aligned SRT generation (optional)
  -> deterministic FFmpeg renderer (optional subtitle burn-in)
  -> output validation and report
```

The LLM must not generate shell commands for direct execution. It may only propose a typed edit plan. All file paths, timestamps, codecs, and output settings are validated before deterministic tools execute.

## Travel-footage safety rule

Silence is not equivalent to dead air. Speech gaps are descriptive metadata only; quiet scenic footage, ambient sound, and music are retained unless the user explicitly requests otherwise.
