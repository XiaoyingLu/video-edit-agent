# AI Video Agent

A local-first Python CLI for editing personal travel footage. Media stays on your machine, source files are never modified, and every edit will be represented by a reviewable plan before rendering.

## MVP

- Folder scanning and media metadata extraction
- Scene detection and representative-frame sampling
- Local vision-model scene understanding with typed shot scale, camera angle, composition, lighting, mood, editorial role, broad setting, and primary subject labels
- Local speech transcription with timestamped segments and long-gap metadata
- Human review of proposed cuts and validated retained-segment timeline
- Timeline-aligned SRT subtitles and optional burned-in subtitles
- Explainable visual highlight candidate detection and ranking
- Natural-language editing intent parsing into a validated, reviewable plan
- Local soundtrack tempo, beat/downbeat, and energy-section analysis

## Hierarchical video planning

The `create-plan` command generates a creative brief, narrative treatment, and structured storyboard using a dedicated lightweight text model. By default it uses `qwen2.5:1.5b`, independently of the multimodal vision model configured by `OLLAMA_MODEL`:

```bash
docker compose run --rm video-agent create-plan ./travel-test \
  --prompt "Create a 90-second cinematic Banff travel film focused on mountain scenery and memorable moments"
```

It writes `.video-agent/video-project-plan.json` and the compatible `.video-agent/editing-plan.json`. The project plan includes a working title, logline, narrative arc, emotional journey, ordered scenes, and concrete storyboard shots with visual descriptions, shot scale, editorial role, duration, narration/text suggestions, and audio cues. Explicit durations in the user's prompt (for example, `90-second` or `2 minutes`) take precedence over model defaults. Scene and shot durations are normalized and validated against the requested total.

Storyboard shots describe intended visuals; they are not claims about footage that has been inspected. Footage paths and source intervals remain unassigned until a later matching stage. The project plan is separate from `.video-agent/optimized-timeline.json`, which contains concrete source paths and time intervals for rendering. Planning does not modify source media, approve edits, or render video. The project-plan artifact depends on the editing-plan artifact, so changing upstream intent invalidates the derived plan. The planning workflow uses the same text model for intent parsing and storyboard generation, evicting it between stages to avoid holding multiple models in memory. Configure it with `VIDEO_PLANNER_MODEL` or override it per command with `create-plan --model`. `VIDEO_PLANNER_MIN_MEMORY_GB` adjusts only the planner's memory preflight; `OLLAMA_MIN_MEMORY_GB` remains available for the general Ollama/vision workflow.

### Storyboard-to-footage suggestions

After generating a storyboard and running `understand`, use `suggest-footage` to rank analyzed source scenes for each planned shot:

```bash
docker compose run --rm video-agent suggest-footage ./travel-test
```

The command writes `.video-agent/footage-suggestions.json`. It compares each shot's purpose and visual description with observed scene descriptions, subjects, objects, activities, shot scale, and editorial role. When the storyboard came from a selected story angle, chapter candidate scenes receive a ranking boost and are labeled as chapter picks; chapter must-include scenes are always retained, even when that makes the candidate list longer than `--max-candidates`. Other footage remains eligible, and a source scene already top-ranked for an earlier shot receives a reuse penalty. Each shot receives ranked candidates with source video, source interval, an explainable score, and matching reasons. Use `--max-candidates` to change the normal candidate limit.

This is a deterministic first-pass matcher, not semantic embedding search or an automatic edit. Candidates remain suggestions: the command does not assign footage to storyboard shots, approve mappings, trim clips, or render video. The report is tracked against both the storyboard and video-understanding artifacts, so changed upstream inputs make the suggestions stale.

## Cinematic scene understanding

The `understand` command analyzes representative frames with the local Ollama vision model and stores both factual descriptions and bounded editorial attributes in `.video-agent/understanding.json`. Each observation includes shot scale, camera angle, composition, lighting, mood, likely editorial role (such as establishing, action, detail, or scenic breath), broad setting, and primary subject. Scene-level story function and dominant mood are aggregated from frame labels. During processing, the CLI reports scene-detection duration, detected scene counts, frame extraction time, per-frame inference time, temporal-analysis time, cache hits, and completed-scene counts. PySceneDetect progress is enabled for interactive CLI runs. This telemetry does not reduce samples, skip temporal analysis, or change model settings.

Understanding is resumable. After each successful frame-analysis request (and after each scene's temporal analysis), progress is atomically saved to `.video-agent/understanding-checkpoint.json`. Rerunning `understand` automatically reuses compatible frame observations and temporal results, so an interruption does not require repeating completed vision work. Extracted frames are also reused when their recorded sample timestamps still match the current sampling settings. The checkpoint is tied to the source-video size/modified time, vision model, and samples-per-scene setting; changing any of these starts a fresh run. Use `understand ./travel-test --no-resume` to deliberately discard saved progress and start over. The checkpoint is removed after the final `understanding.json` has been written successfully.

These are model-assisted visual classifications, not ground truth. The prompt explicitly avoids guessing precise locations or inferring camera movement from a still frame; uncertain classifications are recorded as `unknown`. New fields have defaults so older understanding artifacts remain readable. The attributes are intended to improve later sequence planning and human review; they do not automatically change clip selection or rendering in this step.

## Cinematic sequencing and pacing

The `optimize-timeline` stage uses the scene metadata to build a deterministic visual arc for cinematic and social-media edits. When reliable labels are available, it prefers an establishing view near the opening, varies action/detail/reaction beats through the middle, alternates shot scales where possible, and favors a scenic-breath or establishing shot for the ending. If scene-role metadata is missing or uncertain, it preserves the existing score-ranked order. Vlog and documentary edits, or plans with the `chronological_story` priority, retain source chronology.

Pacing recommendations combine editorial role, inferred subject/camera motion, and (when supplied) soundtrack energy. Stationary scenes can receive more breathing room, while active subject movement and tracking/handheld camera motion can receive tighter targets. Motion adjustments are applied only when the temporal-analysis confidence is at least 0.5; otherwise the role/music target is retained. The combined factors are bounded so uncertain or mixed motion does not dominate the recommendation. The timeline records temporal summaries, motion labels, composition, continuity notes, sequence rationale, and a recommended duration.

Cinematic sequencing also uses temporal action progression when available: clips that follow the same identified subject with a distinct next action receive a continuity preference, while repeated descriptions of the same beat are discouraged. Sequence rationale explains when a shot continues a subject's story. This evidence is carried through reviewed storyboard footage assignments and shown in timeline review. It does not infer events between sampled frames, override chronological-story preferences, or reorder an approved storyboard.

Adjacent-shot continuity checks flag repeated subject/scale/composition combinations and potentially abrupt motion-direction changes. These are review notes only: they never change source intervals or apply trims. All sequencing recommendations remain reviewable, and human approval is still required before rendering.

## Soundtrack assistant

Generate a structured music direction from the creative brief, story treatment, and storyboard:

```bash
docker compose run --rm video-agent suggest-soundtrack ./travel-test
```

This writes `.video-agent/soundtrack-brief.json` with mood and genre directions, instrumentation, a tempo range, an energy arc aligned to the storyboard scenes, catalog search terms, and a hosted-generation prompt. When the storyboard came from a selected story angle, each section preserves its chapter mood and explicit music-energy level; those chapter moods also inform catalog search terms and the generation prompt's timed section directions. Commercial use, monetized-video use, and video synchronization rights are required by default. This is a planning artifact only: it does not download or generate audio, select a track, approve edits, or render video.

Generate original music from the project's soundtrack brief with Stable Audio 3.0. Set `STABILITY_API_KEY` in your shell or `.env` file (never in project JSON or source control):

```bash
docker compose run --rm video-agent generate-soundtrack ./travel-test
```

The command creates an MP3 in `.video-agent/generated-soundtracks/` plus a JSON provenance record containing the generation ID, prompt, checksum, and Stable Audio commercial-use evidence. Use `--format wav` for WAV output or `--seed 42` to request a repeatable seed. Stable Audio 3.0 supports 1–380 seconds per generation; the current API price is 8 credits per successful generation. Generated tracks are not automatically selected, added to the timeline, or rendered. Review each output before publication.

Search the Soundstripe catalog, then create a browser-based audition page. Set `SOUNDSTRIPE_API_KEY` in the environment (never in project JSON or source control):

```bash
docker compose run --rm video-agent search-soundtrack ./travel-test
docker compose run --rm video-agent review-soundtrack ./travel-test \
  --candidates ./.video-agent/soundtrack-candidates.json \
  --video draft.mp4
```

The review page places the optional project-local video beside individual candidate preview players. It displays creative-fit scores, rights status, tags, and provider details. Playback is manual; it does not change the video. Without an account-specific `--license-profile` JSON file confirmed against your Soundstripe agreement, the provider assigns unknown rights and the candidates cannot be selected. Do not set permissions to `true` unless the applicable agreement explicitly confirms them.

Select an eligible candidate by its exact ID from the review JSON:

```bash
docker compose run --rm video-agent select-soundtrack ./travel-test \
  --candidate-id "soundstripe:TRACK_ID"
```

Selection writes `.video-agent/soundtrack-selection.json`. Candidates requiring review or failing the commercial-use, monetization, adaptation, or synchronization gate are blocked. This stage does not download audio, modify the timeline, or render a video.

See [the soundtrack assistant architecture](docs/soundtrack-assistant.md) for provider boundaries, license/provenance requirements, and the incremental implementation plan.

## Music-aware editing

Analyze an explicit soundtrack independently of the video clips:

```bash
docker compose run --rm video-agent analyze-music ./travel-test --music ./travel-test/music/travel-score.mp3
docker compose run --rm video-agent create-plan ./travel-test \
  --prompt "Create a 90-second cinematic mountain film" \
  --music-analysis ./travel-test/.video-agent/music-analysis.json
```

The command decodes audio locally with FFmpeg, computes a spectral-flux onset envelope, estimates tempo with FFT-based autocorrelation, builds a beat grid, and summarizes relative RMS energy in eight-second sections. It writes `.video-agent/music-analysis.json` with estimated BPM, beat and four-beat downbeat timestamps, energy sections, and candidate cut anchors. The soundtrack is recorded as a direct artifact dependency, so replacing the music invalidates the analysis.

This is an advisory analysis stage, not automatic synchronization: beat estimates can be ambiguous for ambient, rubato, syncopated, or rhythmically sparse music. Four-beat downbeats are a practical default rather than inferred time-signature detection. Use beat anchors selectively and preserve natural shot duration; no clip is trimmed, approved, or rendered by this command.

After creating and reviewing a shot timeline, compare its transition points with the soundtrack's beat grid:

```bash
docker compose run --rm video-agent suggest-music-sync ./travel-test --tolerance 0.18
```

The command writes `.video-agent/music-sync-suggestions.json`. It compares hard-cut boundaries and crossfade midpoints with the nearest beat and downbeat, and recommends an anchor only when it is within the configured tolerance and the beat detector has sufficient confidence. The report is review-only: it does not alter clip durations, source intervals, transitions, approvals, or rendered media. The report tracks both the timeline and music analysis as dependencies.

Review the suggestions and create a separate timeline revision. Run without decision flags for an interactive accept/reject/skip review, or pass repeatable `--accept` and `--reject` boundary IDs for a non-interactive review:

```bash
docker compose run --rm -it video-agent review-music-sync ./travel-test

# Or explicitly accept/reject selected boundaries:
docker compose run --rm video-agent review-music-sync ./travel-test \
  --accept shot-001\->shot-002 \
  --reject shot-002\->shot-003
```

Each run saves a new revision and a matching review report under `.video-agent/music-sync-revisions/` (for example, `revision-001.json` and `revision-001.review.json`). The original `.video-agent/optimized-timeline.json` and source media remain unchanged. Accepted shifts are checked against source-media duration and the minimum resulting clip duration; unsafe changes are recorded as infeasible rather than applied. Unlisted boundaries in a flag-based review are skipped.

A music-sync revision is deliberately **not approved**. Inspect and approve that exact revision with the existing timeline review gate, then render the same file:

```bash
docker compose run --rm -it video-agent review-timeline ./travel-test \
  --timeline ./.video-agent/music-sync-revisions/revision-001.json

docker compose run --rm video-agent render-draft ./travel-test \
  --timeline ./.video-agent/music-sync-revisions/revision-001.json
```

Review each clip interactively, or explicitly approve/reject every clip ID shown by the timeline review output. Review decisions are saved to the revision file; rendering remains a separate explicit step and is never started by `review-music-sync`.


## Selective timeline revisions

After reviewing a rendered draft, make explicit, bounded trims to individual shots without replacing the approved source timeline:

```bash
docker compose run --rm video-agent revise-timeline ./travel-test \
  --set-interval shot-0001:12.5:16.0 \
  --set-interval shot-0004:28.0:31.5
```

Each `--set-interval` value is `CLIP_ID:SOURCE_START:SOURCE_END`. A revision can only narrow a shot's existing source interval; it cannot extend beyond the already reviewed footage, change media assignments, or reorder shots. The command saves a separate, versioned file under `.video-agent/timeline-revisions/`. Only edited shots return to pending review, while unchanged approvals are preserved. Transitions are reduced when needed to remain within the renderer's supported overlap limits. The original timeline and source media are never modified.

Inspect the exact changes before approving or rendering:

```bash
docker compose run --rm video-agent compare-timelines ./travel-test \
  --before ./.video-agent/optimized-timeline.json \
  --after ./.video-agent/timeline-revisions/revision-001.json

docker compose run --rm -it video-agent review-timeline ./travel-test \
  --timeline ./.video-agent/timeline-revisions/revision-001.json

docker compose run --rm video-agent render-draft ./travel-test \
  --timeline ./.video-agent/timeline-revisions/revision-001.json \
  --output ./.video-agent/rendered-revision-001.mp4
```

The comparison is read-only and reports changed source intervals, transition adjustments, and total duration delta. A revised timeline remains unapproved until the changed shots are explicitly reviewed; rendering is always a separate command.

## Rendered-draft critique

After rendering and reviewing a draft, inspect it against the creative brief and approved shot timeline:

```bash
docker compose run --rm video-agent critique-draft ./travel-test
```

The command writes `.video-agent/draft-critique.json`. It verifies rendered duration and aspect ratio against the brief, compares output duration with the approved timeline, reports shots already marked outside their pacing targets, samples up to 24 evenly spaced frames for objective luminance/clipping/sharpness measurements, and checks audio-stream presence, average/peak volume, and sustained quiet intervals. Use `--samples` to change the frame sample count, or `--video`, `--timeline`, and `--plan` to select project-local inputs.

For a deeper, optional story check, run:

```bash
docker compose run --rm video-agent critique-draft ./travel-test --semantic --semantic-samples 8
```

Semantic critique uses the configured local Ollama vision model (`OLLAMA_MODEL`, or `--vision-model`) to compare representative rendered shot midpoints with their storyboard purpose, visual description, intended subject, role, and shot scale. It samples at most 12 approved clips, stores each match/partial/mismatch/uncertain assessment with confidence and visual evidence, and adds review findings for confident partial matches or mismatches. This requires the vision model to be available and adds inference time; the default technical-only command does not load a vision model. The assessment is advisory and does not reorder, trim, approve, or render anything.

Findings include the measurements that triggered them. Frame metrics are technical signals, not aesthetic scores: dark scenic footage, intentional silence, and soft-focus shots may all be deliberate creative choices. Review the evidence in context rather than treating a threshold as an automatic defect. The report is registered as depending on the rendered MP4, approved timeline, and creative plan, so changes to those inputs invalidate it. The technical checks are deterministic. Optional semantic checks use the local vision model and are explicitly confidence-scored. Neither mode modifies the timeline or rendered video.

## Compare rendered draft versions

After rendering a revised timeline to a separate MP4, generate a side-by-side review:

```bash
docker compose run --rm video-agent compare-drafts ./travel-test \
  --before ./.video-agent/rendered-draft.mp4 \
  --after ./.video-agent/rendered-revision-001.mp4 \
  --samples 8
```

The command writes a standalone `.video-agent/draft-comparison.html` contact sheet and a machine-readable `.video-agent/draft-comparison.json` report. Each row pairs frames sampled at the same normalized playback position in both videos, even when their total durations differ. The report compares duration, resolution, frame rate, codecs, audio presence/loudness/quiet intervals, and per-frame luminance, near-black/near-white pixel ratios, and sharpness variance. Use `--samples` (1–16) to adjust the number of paired frames, or `--output` to choose another project-local output base path.

The HTML contains embedded frame images, so it can be opened locally without a server or access to the original temporary frame files. The JSON stores metrics and timestamps without duplicating image data. Both reports are registered as depending on the two input MP4s, so changing either render invalidates the comparison. Inputs must be distinct project-local MP4 files.

These measurements are objective signals, not aesthetic scores: higher sharpness or brightness is not inherently better, and normalized positions do not guarantee that two edits show the same scene at the same moment. Inspect both complete videos and listen to the soundtrack before deciding whether to keep a revision. The command never modifies videos, source footage, or timelines.
## Requirements

- Docker Engine and Docker Compose

## Run the complete application with Docker

Both the Python CLI and Ollama/Qwen run in Docker. The CLI container mounts the current workspace at `/workspace`, so your source videos and generated `.video-agent` artifacts remain visible in the Codespace. The two containers communicate over the private Compose network; the CLI reaches Ollama at `http://ollama:11434`.

Build the CLI image and start Ollama:

```bash
docker compose build video-agent
docker compose up -d ollama
```

Download the vision model into the Ollama container (one-time download; model weights persist in a named volume):

```bash
docker compose exec ollama ollama pull qwen2.5vl:3b
docker compose exec ollama ollama pull qwen2.5:1.5b
docker compose exec ollama ollama list
```

For the first Codespaces test, use `qwen2.5vl:3b` for vision and `qwen2.5:1.5b` for planning. On a machine with more available memory, pull `qwen2.5vl:7b` and run vision analysis with `OLLAMA_MODEL=qwen2.5vl:7b`. Planning remains on its separate text model unless `VIDEO_PLANNER_MODEL` is changed.

### Run the CLI

For the end-to-end workflow, use one natural-language command. It analyzes the footage, transcribes speech, ranks highlights, parses your request into a validated plan, selects shots, and builds a proposed timeline. It then asks you to explicitly approve or reject every shot before rendering; if no shot is approved, rendering is skipped. The command never auto-approves clips. It requires an interactive terminal (`-it`) because approval is a deliberate human decision.

```bash
docker compose run --rm -it video-agent create-video ./travel-test --prompt "Create a 60-second cinematic travel video"
```

Tune local resource usage with `--model qwen2.5vl:3b`, `--whisper-model tiny`, or `--samples-per-scene 1`. The workflow writes inspectable artifacts under `.video-agent/`; the individual commands remain available for finer control. Current renderer limitation: it preserves the dimensions of the first approved source shot. Requested aspect ratio/resolution and subtitle/audio preferences are retained in the editing plan but are not yet all applied by this end-to-end command.


Put a short travel clip in a folder in the repository, for example `travel-test/`, then run:

```bash
docker compose run --rm video-agent analyze ./travel-test
docker compose run --rm video-agent understand ./travel-test --samples-per-scene 2
docker compose run --rm video-agent transcribe ./travel-test --model small
docker compose run --rm video-agent detect-silence ./travel-test --noise-db -35 --min-duration 1.5
docker compose run --rm video-agent propose-edits ./travel-test
docker compose run --rm -it video-agent review-edits ./travel-test
docker compose run --rm video-agent build-timeline ./travel-test
docker compose run --rm video-agent find-highlights ./travel-test --samples-per-scene 3
docker compose run --rm video-agent plan ./travel-test --prompt "Create a 60-second cinematic travel video"
docker compose run --rm video-agent draft-edit ./travel-test
docker compose run --rm video-agent optimize-timeline ./travel-test
# Optionally incorporate broad soundtrack energy sections:
docker compose run --rm video-agent optimize-timeline ./travel-test --music-analysis ./travel-test/.video-agent/music-analysis.json
docker compose run --rm -it video-agent review-timeline ./travel-test
docker compose run --rm video-agent subtitles ./travel-test
docker compose run --rm video-agent render ./travel-test --burn-subtitles --subtitle-font-size 28 --subtitle-position bottom
docker compose run --rm video-agent render-draft ./travel-test
```

You can pass any supported CLI arguments after `video-agent`:

```bash
docker compose run --rm video-agent --help
docker compose run --rm -e OLLAMA_MODEL=qwen2.5vl:7b video-agent understand ./travel-test
```

Both `render` and `render-draft` support `--dry-run`. This probes every source with ffprobe (including video dimensions, codecs, pixel format, frame rate, audio sample rate, and channel count), builds the same filter graph used for rendering, then asks FFmpeg to initialize and process a short 0.5-second validation pass to the null muxer. It catches filter syntax and stream-negotiation problems without H.264/AAC encoding or writing an MP4; it is a fast preflight, not a full-duration render guarantee. Example: `docker compose run --rm video-agent render-draft ./travel-test --dry-run`. The `render-draft` command renders the approved `.video-agent/optimized-timeline.json` to `.video-agent/rendered-draft.mp4`. It refuses to run until every proposed shot has an explicit decision and at least one shot is approved. It preserves the reviewed shot order, applies crossfade/dissolve video transitions with matching audio crossfades, uses hard cuts where a transition is not approved or its neighboring shot was rejected, synthesizes silence for clips without audio, probes the finished MP4, and writes it atomically without modifying source footage. It has a separate output and does not change the legacy `render` command. The `optimize-timeline` command turns selected clips into a role-aware cinematic sequence when scene metadata is available, proposes transitions, and assigns adaptive pacing targets by editorial role and shot scale. It writes `.video-agent/optimized-timeline.json`, a review artifact deliberately separate from the renderer's `timeline.json`. Use `review-timeline` to approve or reject each shot interactively, or pass repeated `--approve CLIP_ID` and `--reject CLIP_ID` options. The sequence is approved only after every proposed clip has an explicit decision and at least one clip is approved. Source intervals are preserved exactly; pacing notes are recommendations rather than automatic trims. This stage does not render or alter source media. When soundtrack analysis is available, pass `--music-analysis ./travel-test/.video-agent/music-analysis.json` to `optimize-timeline` to add broad low/medium/high energy context to each proposed shot. Quieter sections suggest longer breathing room, while higher-energy sections suggest tighter pacing; these are advisory targets only. The optimizer preserves source intervals and clip order, and the usual `review-timeline` approval gate still applies. The soundtrack analysis is recorded as an upstream dependency so changes invalidate the optimized timeline. This does not force cuts to beats or infer semantic sections such as choruses or drops. The `plan` command sends a natural-language editing request to the local Ollama model and validates its structured response against a strict Pydantic schema, writing `.video-agent/editing-plan.json`. The `draft-edit` command combines that plan with ranked visual highlights, optional scene descriptions, and optional transcript excerpts to produce a duration-bounded, reviewable shot selection in `.video-agent/draft-selection.json`. Selections are recommendations only: the command does not change the approved timeline or render a video. The `find-highlights` command detects scenes, measures sampled-frame motion, sharpness, exposure, and saturation, then writes ranked, explainable candidates to `.video-agent/highlights.json`. The score is a deterministic visual heuristic, not an aesthetic judgment; every candidate remains review-only, and low-motion scenic footage is explicitly preserved as a possible highlight. The `analyze` command scans the folder and extracts media metadata using ffprobe. The `understand` command detects scenes, extracts representative frames, sends each frame to the local Ollama model, and writes `.video-agent/understanding.json`. For scenes with at least two sampled frames, it also sends an ordered multi-frame sequence to the local vision model to estimate temporal action progression, subject movement, and camera movement. Motion labels are conservative (unknown when uncertain), and temporal-analysis failures produce a warning while preserving the per-frame scene analysis. These temporal descriptions become additional evidence for lexical and embedding-based storyboard footage matching. The `transcribe` command uses faster-whisper locally to recognize speech from video audio streams and writes `.video-agent/transcription.json`. The `detect-silence` command runs FFmpeg’s `silencedetect` filter and writes `.video-agent/silence-analysis.json`, correlating quiet intervals with transcript timestamps when available. The `propose-edits` command combines those review candidates with optional scene descriptions and writes a human-reviewable `.video-agent/edit-proposal.json`. Use `review-edits` to approve/reject each item interactively, or pass repeated `--approve ITEM_ID` and `--reject ITEM_ID` options for scripted review. Skipped items remain pending. Once every item has an explicit decision, `build-timeline` writes `.video-agent/timeline.json` containing the ordered source intervals to retain. This validated timeline is an edit-decision artifact only. After every cut has been reviewed, `render` uses FFmpeg to create `.video-agent/rendered.mp4` as an H.264/AAC MP4. The renderer checks the timeline and source files, keeps all source media untouched, refuses to overwrite an existing output unless `--overwrite` is supplied, and probes the finished MP4 before reporting success. Clips with no audio receive generated silence so all segments can be concatenated consistently. The subtitles command reads the Whisper transcription and validated timeline, remaps speech captions through removed intervals and concatenated clips, and writes .video-agent/subtitles.srt. Render with --burn-subtitles to burn the SRT into the final output after concatenation; use --subtitle-font-size (12–72) and --subtitle-position (bottom, center, top) to adjust its appearance, or pass --subtitle-file for a custom SRT. The `--subtitle-style` presets are `classic` (the unchanged default), `cinematic` (strong outline and larger safe margin), `boxed` (translucent caption panel), and `minimal` (lighter outline). Non-classic presets scale font size and vertical safe margins against the output height, including portrait exports.

Transcription defaults to the CPU-friendly `small` model with `int8` compute. The first run downloads Whisper model weights into `.video-agent/models/`; subsequent runs reuse the cache. Use `--model tiny` for a quicker smoke test or `--device cuda --compute-type float16` on a compatible NVIDIA setup. Videos without audio are recorded in the output with an empty segment list and are not sent to Whisper.

Example output layout:

```text
travel-test/
└── .video-agent/
    ├── analysis.json
    ├── understanding.json
    ├── transcription.json
    ├── silence-analysis.json
    ├── edit-proposal.json
    ├── timeline.json
    ├── highlights.json
    ├── draft-selection.json
    ├── optimized-timeline.json
    ├── music-analysis.json
    ├── subtitles.srt
    ├── rendered.mp4
    ├── models/
    └── frames/
        └── <video-hash>/scene-*/frame-*.jpg
```

Each transcript segment contains start/end timestamps, recognized text, and optional Whisper confidence diagnostics. The artifact also reports long gaps before, between, and after speech segments. These gaps are descriptive only—not automatic cut suggestions. Silence analysis reports low-volume intervals and emits review candidates only when a transcript is available and no speech overlaps the interval (with a configurable safety margin). Quiet intervals may still contain scenic ambience, music, wind, or other intentional sound: candidates are never automatic cuts and always require human review. Original video files are not modified.

### Configuration and resources

The Compose service defaults to `qwen2.5vl:3b`. Override it for a command with `-e OLLAMA_MODEL=...`. Ollama is configured for one loaded model and one parallel inference request. Intent parsing and frame understanding explicitly unload their model when each phase ends; the end-to-end workflow therefore releases Qwen before starting CPU-heavy Whisper transcription. Whisper releases its CTranslate2 model references and runs garbage collection when transcription completes. These are best-effort release operations and do not discard successful artifacts if the Ollama unload request fails.

Before loading a model, the CLI checks available host/container memory against conservative model-size estimates. Whisper CUDA mode also checks free NVIDIA VRAM when `nvidia-smi` is available. If memory cannot be measured, the check is skipped rather than blocking execution. These estimates are guardrails, not guarantees: actual usage depends on quantization, context size, concurrent applications, and backend configuration. In the default Compose setup, containers share the host memory pool; if you add separate memory limits to the Ollama service, ensure the configured budget reflects that service's limit.

Default estimates are 6 GB for Qwen 2.5 VL 3B, 12 GB for 7B, 3 GB for Whisper small, and 3 GB VRAM for Whisper small on CUDA. Adjust them when your hardware/model quantization differs:

```bash
# Lower or raise the preflight thresholds in GB (use only when you know the real budget).
OLLAMA_MIN_MEMORY_GB=9 WHISPER_MIN_MEMORY_GB=2 WHISPER_MIN_VRAM_GB=2 \
  docker compose run --rm video-agent transcribe ./travel-test --model small

# Change how long Ollama keeps a model resident between requests.
# Put this in your shell or .env file before starting services and running commands.
export OLLAMA_KEEP_ALIVE=2m
docker compose up -d ollama
```

Ollama's API is also published on loopback port 11434 for optional diagnostics:

```bash
curl http://127.0.0.1:11434/api/tags
docker compose logs -f ollama
```

Qwen vision models and Whisper weights require disk and memory; inference may be slow on CPU-only Codespaces. Start with a short 30–90 second clip, one or two frames per scene, and Whisper's `tiny` model if needed. The default Compose setup makes no GPU assumption. NVIDIA acceleration requires NVIDIA Container Toolkit and an appropriate Compose GPU configuration.

### Stop services

```bash
docker compose down
```

Downloaded Ollama weights remain in the named volume, and Whisper weights remain in the workspace cache. To stop services and delete the Ollama model volume too (destructive):

```bash
docker compose down -v
```

## Development

The Docker image includes the development tools. Run tests and lint without starting Ollama:

```bash
docker compose build video-agent
docker compose run --rm --no-deps --entrypoint python video-agent -m pytest
docker compose run --rm --no-deps --entrypoint ruff video-agent check .
docker compose run --rm --no-deps --entrypoint ruff video-agent format --check .
```

For a non-Docker development environment, install the development and transcription extras:

```bash
python -m pip install -e ".[dev,transcription]"
pytest
ruff check .
ruff format --check .
```

## Artifact provenance and stale-cache protection

The CLI records generated artifacts in `.video-agent/artifact-manifest.json`. Each tracked artifact stores its SHA-256, the source-video inventory and sampled content fingerprints, direct upstream dependencies, and relevant generation settings. Dependency checks recurse upstream, so a changed source video or modified upstream JSON artifact invalidates downstream results before they are reused or rendered.

Check the project at any time:

```bash
docker compose run --rm video-agent validate-artifacts ./travel-test
```

### Generated storage cleanup

Extracted frames are regenerable caches and can consume substantial disk space on long trips. The `clean` command prunes only files under `.video-agent/frames/` and `.video-agent/highlight-frames/`: it first removes files older than seven days, then removes the oldest remaining cache files until the combined cache is at or below 5 GB. Both limits are configurable.

Preview the operation before deleting anything:

```bash
docker compose run --rm video-agent clean ./travel-test --dry-run
```

Apply it, optionally preserving frames referenced by the current `understanding.json`:

```bash
docker compose run --rm video-agent clean ./travel-test --max-age-days 7 --max-cache-gb 5
docker compose run --rm video-agent clean ./travel-test --keep-artifacts
```

The cleanup never removes source videos, JSON analysis/review artifacts, the artifact manifest, SRT files, rendered MP4s, or downloaded model weights. `--keep-artifacts` additionally protects frame paths referenced by `understanding.json`. Use `--dry-run` to inspect the candidate count and reclaimed-space estimate first.

The command reports stale artifacts and exits non-zero when regeneration is required. The end-to-end `create-video` workflow and the standalone understanding, transcription, highlight, planning, selection, timeline, and rendering commands register their outputs. Legacy artifacts created before dependency tracking remain readable for compatibility, but are not validated because they have no manifest entry. Regenerate them with the current CLI to gain provenance checks.

Source files are never modified. File size and modification time provide a fast check; a sampled SHA-256 over the beginning, middle, and end detects content changes even when timestamps are preserved, and a full SHA-256 is recomputed when the sample or file metadata changes. The manifest is local project metadata and is covered by the existing `.video-agent/` Git ignore rule.

## Safety and privacy

- Original media is read-only from this application. Rendering writes to a separate output file and never uses a source path as its destination.
- Legacy rendering is gated on a validated timeline with no pending review items. Draft rendering additionally requires every proposed shot to be explicitly approved or rejected, with at least one approved shot. Existing MP4 outputs are not replaced unless `--overwrite` is explicitly supplied.
- Analysis artifacts, extracted frames, and local model caches are stored under `.video-agent/` and ignored by Git.
- Video/audio media, generated renders, and environment files are ignored by Git.
- Do not commit personal footage, generated renders, API keys, or private transcripts.
- Vision and transcription outputs are descriptive evidence, not editing instructions. The future LLM planner must produce a validated edit-plan data structure and must never execute arbitrary model-generated shell commands.

## Roadmap

1. Media scanning, scene detection, representative-frame sampling, and visual descriptions
2. Speech transcription and speech-bounded silence detection
3. Motion understanding, explainable highlight ranking, and quality assessment (initial visual-signal MVP implemented)
4. Validated edit-plan generation and dry-run review
5. Deterministic FFmpeg rendering and output validation (implemented)
6. Timeline-aligned SRT generation and optional burned-in subtitles (implemented)
7. Natural-language request parsing into a validated editing plan (implemented)
8. Context-aware clip selection (implemented)
9. Cinematic scene metadata, role-aware shot sequencing, adaptive pacing targets, transition proposals, and human review (implemented)
10. Render approved shot timelines with FFmpeg transitions and output verification (implemented; Docker validation pending)
11. Music-aware soundtrack tempo, beat/downbeat, and energy-section analysis, plus review-only suggestions for aligning timeline transition points to nearby musical anchors


### Compile the approved storyboard into a production timeline

Once every storyboard shot has an explicitly approved footage assignment, approve the creative storyboard and compile it into the existing reviewable FFmpeg timeline:

```bash
docker compose run --rm video-agent approve-storyboard ./travel-test
docker compose run --rm video-agent build-storyboard-timeline ./travel-test
docker compose run --rm video-agent review-timeline ./travel-test
docker compose run --rm video-agent render-draft ./travel-test
```

The compilation preserves storyboard scene/shot order and uses the approved source intervals. If an assigned source interval is longer than its shot target, the compiler takes a centered trim; if it is shorter, it preserves the available interval without stretching. Transition preferences come from the validated editing plan, with the first shot always beginning on a cut. Every compiled timeline clip starts as `pending`; the normal `review-timeline` approval gate still applies before rendering. The renderer remains deterministic and does not modify source media.

### Visual storyboard review

After creating the storyboard and analyzing source footage, generate a local HTML review page:

```bash
docker compose run --rm video-agent review-storyboard ./travel-test
```

Open `.video-agent/storyboard.html` in a browser. The page groups shots by narrative scene and shows each shot's purpose, visual description, target duration, assigned source interval, editorial role, and a representative analyzed frame when one is available. It is a read-only review artifact: it does not approve the creative plan, change footage assignments, or render a video. Re-run it after changing the plan or footage mappings.

### Review storyboard footage matches

After generating a creative storyboard and analyzing the source videos, rank candidate scenes:

```bash
docker compose run --rm video-agent suggest-footage ./travel-test
```

The default matcher uses explainable lexical overlap plus shot-scale and editorial-role evidence. For stronger semantic retrieval, install a local Ollama embedding model and opt in:

```bash
docker compose exec ollama ollama pull nomic-embed-text
docker compose run --rm video-agent suggest-footage ./travel-test --embedding-model nomic-embed-text
```

Hybrid ranking combines embedding similarity with lexical coverage and visual-role/shot-scale matches. The report records the model, semantic similarity, lexical score, and matching method for each candidate. Embeddings run locally through Ollama; without `--embedding-model`, the existing lexical matcher is used. Matches remain suggestions only and still require explicit human review.

Inspect `.video-agent/footage-suggestions.json`, then explicitly approve candidate numbers per shot (candidate numbers are 1-based):

```bash
docker compose run --rm video-agent review-footage ./travel-test \
  --select shot-001=1 \
  --select shot-002=2
```

Repeat `--select SHOT_ID=CANDIDATE_NUMBER` for each shot. Use `--reject SHOT_ID` to clear an earlier assignment. Unselected shots remain unassigned. The project advances to `footage_mapped` only when every storyboard shot has an explicitly approved source interval. This stage does not approve the overall creative plan, alter source media, or render a video. Since chapter must-include scenes are retained as candidates for every shot in that chapter, selecting the same source more than once can create repeated footage. `review-footage` and timeline compilation emit a warning when approved source intervals overlap; reuse is allowed and must be judged intentionally. This does not block approval or compilation.


### Declarative animated title overlays

Create a title-overlay review artifact after building the optimized timeline:

```bash
docker compose run --rm video-agent init-title-overlays ./travel-test
```

Edit `.video-agent/title-overlays.json` and add entries to its `overlays` array. Each entry has a unique ID, text (maximum 120 characters), output-time start/end, placement, built-in style, and animation preset. Example:

```json
{
  "overlay_id": "opening-title",
  "text": "A Weekend in the Rockies",
  "start_seconds": 0.5,
  "end_seconds": 4.5,
  "placement": "bottom",
  "style": "cinematic",
  "animation": "fade",
  "decision": "pending"
}
```

Supported placements are `top`, `center`, and `bottom`; styles are `cinematic`, `minimal`, and `bold`; animations are `none`, `fade`, and `slide_up`. These map to fixed renderer presets—title data cannot supply arbitrary FFmpeg filters, expressions, fonts, or commands. Title times refer to the final concatenated output, not source-video timestamps.

Review each title explicitly, then render only approved overlays:

```bash
docker compose run --rm video-agent review-title-overlays ./travel-test --approve opening-title
docker compose run --rm video-agent render-draft ./travel-test --title-overlays ./travel-test/.video-agent/title-overlays.json --dry-run
docker compose run --rm video-agent render-draft ./travel-test --title-overlays ./travel-test/.video-agent/title-overlays.json
```

Rejected and pending overlays are ignored. Overlay intervals must fit within the approved rendered duration. The renderer applies titles after shot transitions, so title timing is stable on the final output timeline. The title artifact tracks the optimized timeline as an upstream dependency; source clips and the reviewed shot timeline remain unchanged. The same artifact also supports cinematic lower thirds. Set `overlay_kind` to `lower_third`; use `text` for the location/person name and optional `secondary_text` for a date or short context line. The renderer uses a fixed, percentage-based lower-left card with safe margins that adapt to landscape, portrait, and square output. Both lines are escaped and rendered with expansion disabled. Lower thirds use the same bounded fade/slide presets and remain pending until explicitly approved.

### Narrative-aware opening and closing title suggestions

After creating a creative video plan and approving the optimized timeline, generate an editable title-sequence proposal:

```bash
docker compose run --rm video-agent suggest-title-sequences ./travel-test
```

The command uses the story treatment's working title and logline for the opening, and the final narrative-arc beat for the closing card. Timing is derived from the approved timeline's actual duration, including transition overlaps. It refuses to run without an approved timeline and a story treatment, and it never overwrites an existing title review artifact.

The generated `.video-agent/title-overlays.json` contains three pending suggestions: an opening title, an opening logline, and a closing card. Each is marked with its `sequence_role` and uses existing fixed animation/style presets. These are suggestions, not approved edits. Review the exact entries you want to use, then render:

```bash
docker compose run --rm video-agent review-title-overlays ./travel-test --approve opening-title --approve opening-logline --approve closing-card
docker compose run --rm video-agent render-draft ./travel-test --title-overlays ./travel-test/.video-agent/title-overlays.json --dry-run
docker compose run --rm video-agent render-draft ./travel-test --title-overlays ./travel-test/.video-agent/title-overlays.json
```

The title artifact records dependencies on both the creative plan and optimized timeline. Regenerate suggestions to a separate output path if you want to preserve an existing review. No footage is reordered or modified, and no title is automatically approved.

### Output aspect ratio and resolution

When a current `.video-agent/video-project-plan.json` exists, `render-draft` uses its requested aspect ratio and resolution (16:9, 9:16, or 1:1; 720p, 1080p, or 4K). The rendered dimensions are verified with ffprobe, and the render artifact tracks the project plan as an upstream dependency.

Use `--fit-mode crop` (default) to fill the requested frame with a centered crop, or `--fit-mode pad` to preserve the complete source frame with letterboxing/pillarboxing. Crop can remove content near the edges; pad preserves all source content. If no project plan exists, the first approved source clip continues to determine output dimensions.

```bash
docker compose run --rm video-agent render-draft ./travel-test --fit-mode pad
```


### Cinematic lower thirds

Add a lower-third entry to `.video-agent/title-overlays.json` alongside title cards:

```json
{
  "overlay_id": "lake-louise-location",
  "overlay_kind": "lower_third",
  "text": "Lake Louise",
  "secondary_text": "Banff National Park · Alberta",
  "start_seconds": 18.0,
  "end_seconds": 23.0,
  "animation": "fade",
  "decision": "pending"
}
```

Use the primary line for a place or person name and the optional secondary line for a date or concise description. The fixed lower-left card uses proportional safe-area geometry for 16:9, 9:16, and 1:1 outputs; it does not accept custom coordinates, fonts, FFmpeg expressions, or filter text. Review it like any other title overlay before rendering:

```bash
docker compose run --rm video-agent review-title-overlays ./travel-test --approve lake-louise-location
docker compose run --rm video-agent render-draft ./travel-test --title-overlays ./travel-test/.video-agent/title-overlays.json --dry-run
```


### Subtitle presentation presets

Burn captions into a rendered timeline with a fixed appearance preset:

```bash
docker compose run --rm video-agent render ./travel-test --burn-subtitles --subtitle-style cinematic
docker compose run --rm video-agent render ./travel-test --burn-subtitles --subtitle-style boxed --subtitle-position bottom
docker compose run --rm video-agent render ./travel-test --subtitle-file ./travel-test/captions.srt --subtitle-style minimal --dry-run
```

Available styles are:

- `classic`: the existing white text with dark outline and shadow; this remains the default and preserves its previous sizing and placement.
- `cinematic`: bold white captions with a stronger outline and generous safe-area margin.
- `boxed`: white captions on a translucent dark panel.
- `minimal`: a lighter outline and shadow for a restrained look.

For non-classic presets, font size and vertical margin scale with output height, so portrait exports receive larger captions and proportionate bottom clearance. `--subtitle-font-size` remains the 1080p reference size (12–72); `--subtitle-position` supports bottom, center, or top. Subtitle cue timestamps and text are unchanged by presentation styling. The renderer uses only built-in presets and does not accept arbitrary ASS/FFmpeg style strings.


### Critique-driven shot revision suggestions

After generating a rendered-draft critique, turn its confident semantic concerns and approved-timeline pacing outliers into a shot-level review list:

```bash
docker compose run --rm video-agent suggest-revisions ./travel-test --minimum-confidence 0.55
```

The command writes `.video-agent/draft-revision-suggestions.json`. Semantic mismatches and partial matches are mapped to their existing storyboard/timeline shot IDs only when confidence meets the threshold. Pacing outliers are surfaced from the approved timeline. Each suggestion includes its evidence, affected shot, current source interval, and a bounded review action.

These are review prompts, not generated edits. The tool does not invent trim timestamps, extend source handles, replace footage, reorder shots, approve a timeline, or render a video. Use the existing `revise-timeline --set-interval` workflow only after choosing an exact, safe trim yourself; footage reassignment remains a separate storyboard review workflow. The suggestion artifact tracks both the critique report and source timeline, so stale upstream inputs must be regenerated.


To review suggestions interactively:

```bash
docker compose run --rm video-agent review-revision-suggestions ./travel-test
```

Choose accept, reject, or skip for each item. For an accepted overlong timing suggestion, you can enter an exact `start:end` source interval; the tool validates that it is a trim within the existing source range. Alternatively, use repeatable flags for a scripted review:

```bash
docker compose run --rm video-agent review-revision-suggestions ./travel-test \
  --accept shot-b-pacing --reject shot-a-story \
  --set-interval shot-b:6.0:9.0
```

The review decisions are saved under `.video-agent/revision-reviews/`. If an accepted overlong shot has an explicit interval, a separate unapproved timeline is created under `.video-agent/timeline-revisions/`. The original approved timeline is never overwritten. Accepted footage-review suggestions are recorded as follow-up tasks; they do not silently replace footage. Short shots cannot be extended by this workflow.


To inspect a timeline revision visually before approving or rendering it:

```bash
docker compose run --rm video-agent review-timeline-revision ./travel-test \
  --before .video-agent/optimized-timeline.json \
  --after .video-agent/timeline-revisions/revision-001.json
```

The command writes a JSON report and a self-contained HTML page under
`.video-agent/timeline-revision-review/`. For every changed shot, the page pairs a
representative midpoint frame from the original source interval with one from the revised
interval and shows both source ranges. It does not render the timeline or modify either
timeline or any source media. These are source-frame samples, not a substitute for watching
the full revised sequence and checking transition context.


For a dedicated revision-preview workflow, render an approved revision to a separate, non-overwriting output:

```bash
docker compose run --rm video-agent render-revision-preview ./travel-test \
  --timeline ./.video-agent/timeline-revisions/revision-001.json
```

The command requires the revision to be fully approved with `review-timeline`. It writes
`.video-agent/revision-previews/revision-001.mp4` and a matching
`revision-001.preview.json` manifest containing the source timeline, output path, duration,
resolution, codecs, and shot count. Use `--dry-run` to validate the FFmpeg graph without
writing a preview, or `--output` to choose another project-local MP4 path. Existing preview
files and manifests are never overwritten. The preview and manifest are tracked against the
approved revision; the original timeline and source media remain unchanged.

Compare the new preview with the earlier render using `compare-drafts` as described above.


## Revision iteration reports

After reviewing and rendering a timeline revision, create a single index that connects the evidence from the full iteration:

```bash
docker compose run --rm video-agent report-revision-cycle ./travel-test \
  --critique ./.video-agent/draft-critique.json \
  --suggestions ./.video-agent/draft-revision-suggestions.json \
  --original-timeline ./.video-agent/optimized-timeline.json \
  --revised-timeline ./.video-agent/timeline-revisions/revision-001.json \
  --preview-manifest ./.video-agent/revision-previews/revision-001.preview.json \
  --comparison ./.video-agent/draft-comparison.json \
  --timeline-review ./.video-agent/timeline-revision-review/review.json
```

The command writes a JSON index and a self-contained HTML report under
`.video-agent/revision-cycles/`. It validates that every supplied artifact is project-local,
tracked, and current, then links to the original critique, suggestions, timelines, preview
manifest, and rendered comparison. The source-frame timeline review is optional. The report
does not infer approval, change media, or replace the detailed evidence reports; it provides a
traceable entry point for reviewing one iteration. Re-run with a different `--output` base
for subsequent cycles; existing reports are never overwritten.



## Browser-based music and timeline synchronization review

After analyzing the soundtrack and generating beat-alignment suggestions, create a shared-time-axis preview:

```bash
docker compose run --rm video-agent review-music-timeline ./travel-test
```

Open `.video-agent/music-timeline-preview/preview.html`. The page includes an audio player for the analyzed soundtrack, broad energy sections, estimated beat/downbeat markers, shot intervals, current transition boundaries, and suggested beat anchors. The playhead follows audio playback, the active shot is highlighted, and clicking the timeline or using the left/right arrow keys seeks the soundtrack. All tracks use seconds on the same horizontal time axis. The audio file is referenced in place; it is not copied into the report.

Generate or refresh the inputs first when needed:

```bash
docker compose run --rm video-agent analyze-music ./travel-test --music ./travel-test/music.mp3
docker compose run --rm video-agent suggest-music-sync ./travel-test
docker compose run --rm video-agent review-music-timeline ./travel-test
```

Beat and downbeat positions are signal-based estimates, not ground truth. Listen to the soundtrack and inspect the footage; not every transition should be forced onto a beat. This browser page is read-only and does not alter or approve the timeline. To accept or reject proposed timing changes, use the existing `review-music-sync` workflow, which writes a separate unapproved timeline revision.

## Browser-based visual timeline review

For a visual review of the proposed shot order, generate an offline HTML page with one
representative midpoint frame per shot and its scene, motion, continuity, pacing, music, and
transition context:

```bash
docker compose run --rm video-agent review-timeline-visual ./travel-test
```

Open `.video-agent/timeline-visual-review/review.html` in a browser. The built-in player can
play the current shot or the complete sequence in card order, honoring each shot's source in/out
points. Use Previous/Next to inspect individual shots. The preview reads the original local video
files through paths relative to the HTML page; keep the review page in its generated project
folder. If your browser blocks local media access, serve the project folder on loopback:

```bash
cd ./travel-test
python3 -m http.server 8000 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8000/.video-agent/timeline-visual-review/review.html`.
Playback uses cuts between clips, so it is a sequence/pacing preview rather than a final
transition-accurate render.

Drag cards to change shot order and adjust the source in/out points to trim a shot. Use
**Export visual edits (.json)** to save a proposal, then import it as a separate revision:

```bash
docker compose run --rm video-agent apply-timeline-visual-edits ./travel-test \
  --edits-file ./.video-agent/timeline-visual-edits.json
```

The CLI writes a new unapproved timeline under `.video-agent/timeline-revisions/`. It requires
the export to contain every original clip exactly once, rejects stale timeline hashes and trims
outside each shot's existing source interval, and enforces a one-second minimum shot duration.
Because reordering changes shot adjacency, existing transitions are reset to cuts for review.
The original timeline and source media remain unchanged. Review the new revision with
`review-timeline --timeline PATH` before rendering.

The page also retains **Export review decisions (.json)** for the original approval workflow.
Move that file into the project folder and apply it through the existing gate:

```bash
docker compose run --rm video-agent review-timeline ./travel-test \
  --decisions-file ./.video-agent/timeline-review-decisions.json
```

Both exports are bound to the exact timeline file hash. Approval decisions are not applied by
the browser itself; the CLI validates and applies them. Source videos are read only for frame
extraction and are never modified.


## Compare generated and catalog soundtracks

After creating a soundtrack brief and generating one or more Stable Audio tracks, prepare a unified candidate collection. This analyzes each generated audio file locally with the existing FFmpeg/NumPy music analyzer and combines it with saved Soundstripe results when present:

```bash
docker compose run --rm video-agent prepare-soundtrack-candidates ./travel-test
docker compose run --rm video-agent review-soundtrack ./travel-test \
  --candidates ./.video-agent/soundtrack-candidates-combined.json \
  --video ./travel-test/draft.mp4
```

Generated candidates retain their provider generation ID, license evidence, checksum, local asset path, estimated tempo, and measured duration. The read-only HTML audition page plays generated files directly from the project and catalog previews from their provider URLs. Review the rights status and listen before selecting an eligible candidate:

```bash
docker compose run --rm video-agent select-soundtrack ./travel-test \
  --candidate-id stable-audio:GENERATION_ID
```

This workflow only analyzes and compares audio and records an explicit selection. It does not change the edit timeline or render a video. If a browser blocks local audio playback, serve the project folder on loopback with `python3 -m http.server 8000 --bind 127.0.0.1` and open the review HTML through that local server.


## Analyze the selected soundtrack

After explicitly selecting an eligible candidate, analyze that exact track to create the standard `.video-agent/music-analysis.json` artifact consumed by `suggest-music-sync`:

```bash
# Generated Stable Audio candidate: uses its recorded project-local audio file.
docker compose run --rm video-agent analyze-selected-soundtrack ./travel-test

# Soundstripe/catalog candidate: first save the licensed audio using your account's
# authorized download workflow inside the project, then provide that local file.
docker compose run --rm video-agent analyze-selected-soundtrack ./travel-test \
  --music ./.video-agent/licensed-assets/selected-track.mp3

docker compose run --rm video-agent suggest-music-sync ./travel-test
```

The command validates the human selection, current review, rights eligibility, project-local audio path, and generated-track checksum. Catalog preview URLs are not treated as downloadable or renderable assets; catalog audio must be supplied as a local file obtained under the applicable account license. It records analysis and provenance dependencies without changing the timeline or rendering.


### Build the storyboard from a selected story angle

After generating story alternatives and explicitly selecting one, pass the proposal to `create-plan`:

```bash
docker compose run --rm video-agent propose-story ./travel-test --prompt "relaxed, 90 seconds"
docker compose run --rm video-agent select-story ./travel-test --angle angle-1
docker compose run --rm video-agent create-plan ./travel-test \\
  --prompt "Create a relaxed cinematic travel film" \\
  --story ./.video-agent/story-proposal.json
```

The selected angle supplies the chapter order, names, purposes, moods, narration hints, music-energy intent, candidate source scene IDs, and chapter target durations. The planner generates concrete storyboard shots within each chapter and preserves the chapter's target duration. The story proposal must be current, belong to the same project, and have an explicitly selected angle. If the prompt specifies a duration, it must match the proposal's duration.

The resulting plan remains a storyboard proposal: candidate source scenes are carried forward as hints for footage review, but no footage is assigned or approved and no timeline is changed or rendered. The plan tracks the story proposal as an upstream artifact, so changing or regenerating the proposal makes the plan stale.

### Music-section-aware storyboard planning

When creating a creative video plan, optionally pass the selected soundtrack's current `.video-agent/music-analysis.json` artifact:

```bash
docker compose run --rm video-agent create-plan ./travel-test \
  --prompt "Create a cinematic travel film" \
  --music-analysis ./.video-agent/music-analysis.json
```

The planner receives the broad low/medium/high energy sections as creative context, then the application annotates every scene and storyboard shot against its actual output-time interval. Each annotation records the dominant energy, the ordered energy progression when an interval crosses multiple sections, the matching section bounds, and an explanatory note. This helps distinguish a quiet scenic hold from a build or high-energy passage without pretending the analyzer has identified a chorus, drop, or musical phrase.

Energy annotations are advisory: they do not change shot durations, force cuts onto beats, reorder footage, approve the storyboard, modify a timeline, or render. The generated plan tracks the exact music-analysis artifact as a dependency; regenerate the plan if the analysis or selected soundtrack changes.


### Music-aware pacing and transition suggestions

When `create-plan` receives `--music-analysis`, each storyboard shot also receives advisory `pacing_recommendation` and `transition_recommendation` fields with short rationales. Low-energy scenic passages can suggest a linger; energetic action passages can suggest a brisk rhythm; adjacent shots with compatible composition or visible motion can suggest a match cut or cut on action. Other cases use a clean cut or a gentle dissolve where an energy change supports it. If no music section covers a shot, the planner falls back to the brief's overall pacing preference.

These are editorial suggestions, not edit instructions: they never change shot durations, add transition effects to the timeline, snap cuts to beats, approve footage, or trigger rendering. The final storyboard and timeline remain subject to human review.
