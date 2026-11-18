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

The command writes `.video-agent/footage-suggestions.json`. It compares each shot's purpose and visual description with observed scene descriptions, subjects, objects, activities, shot scale, and editorial role. Each shot receives up to three ranked candidates with source video, source interval, an explainable score, and matching reasons. Use `--max-candidates` to change the number of candidates.

This is a deterministic first-pass matcher, not semantic embedding search or an automatic edit. Candidates remain suggestions: the command does not assign footage to storyboard shots, approve mappings, trim clips, or render video. The report is tracked against both the storyboard and video-understanding artifacts, so changed upstream inputs make the suggestions stale.

## Cinematic scene understanding

The `understand` command analyzes representative frames with the local Ollama vision model and stores both factual descriptions and bounded editorial attributes in `.video-agent/understanding.json`. Each observation includes shot scale, camera angle, composition, lighting, mood, likely editorial role (such as establishing, action, detail, or scenic breath), broad setting, and primary subject. Scene-level story function and dominant mood are aggregated from frame labels.

These are model-assisted visual classifications, not ground truth. The prompt explicitly avoids guessing precise locations or inferring camera movement from a still frame; uncertain classifications are recorded as `unknown`. New fields have defaults so older understanding artifacts remain readable. The attributes are intended to improve later sequence planning and human review; they do not automatically change clip selection or rendering in this step.

## Cinematic sequencing and pacing

The `optimize-timeline` stage uses the scene metadata to build a deterministic visual arc for cinematic and social-media edits. When reliable labels are available, it prefers an establishing view near the opening, varies action/detail/reaction beats through the middle, alternates shot scales where possible, and favors a scenic-breath or establishing shot for the ending. If scene-role metadata is missing or uncertain, it preserves the existing score-ranked order. Vlog and documentary edits, or plans with the `chronological_story` priority, retain source chronology.

Pacing recommendations are now role-aware: establishing and scenic-breath shots receive more room, while action and detail shots receive shorter targets. The timeline records each shot's editorial role, scale, sequence rationale, and recommended duration. These targets are advisory only: source intervals remain unchanged, and the reviewer decides whether to trim or extend footage. The ordering is deterministic and reviewable; it does not auto-approve shots or render media.

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

Both `render` and `render-draft` support `--dry-run`. This probes every source with ffprobe (including video dimensions, codecs, pixel format, frame rate, audio sample rate, and channel count), builds the same filter graph used for rendering, then asks FFmpeg to initialize and process a short 0.5-second validation pass to the null muxer. It catches filter syntax and stream-negotiation problems without H.264/AAC encoding or writing an MP4; it is a fast preflight, not a full-duration render guarantee. Example: `docker compose run --rm video-agent render-draft ./travel-test --dry-run`. The `render-draft` command renders the approved `.video-agent/optimized-timeline.json` to `.video-agent/rendered-draft.mp4`. It refuses to run until every proposed shot has an explicit decision and at least one shot is approved. It preserves the reviewed shot order, applies crossfade/dissolve video transitions with matching audio crossfades, uses hard cuts where a transition is not approved or its neighboring shot was rejected, synthesizes silence for clips without audio, probes the finished MP4, and writes it atomically without modifying source footage. It has a separate output and does not change the legacy `render` command. The `optimize-timeline` command turns selected clips into a role-aware cinematic sequence when scene metadata is available, proposes transitions, and assigns adaptive pacing targets by editorial role and shot scale. It writes `.video-agent/optimized-timeline.json`, a review artifact deliberately separate from the renderer's `timeline.json`. Use `review-timeline` to approve or reject each shot interactively, or pass repeated `--approve CLIP_ID` and `--reject CLIP_ID` options. The sequence is approved only after every proposed clip has an explicit decision and at least one clip is approved. Source intervals are preserved exactly; pacing notes are recommendations rather than automatic trims. This stage does not render or alter source media. The `plan` command sends a natural-language editing request to the local Ollama model and validates its structured response against a strict Pydantic schema, writing `.video-agent/editing-plan.json`. The `draft-edit` command combines that plan with ranked visual highlights, optional scene descriptions, and optional transcript excerpts to produce a duration-bounded, reviewable shot selection in `.video-agent/draft-selection.json`. Selections are recommendations only: the command does not change the approved timeline or render a video. The `find-highlights` command detects scenes, measures sampled-frame motion, sharpness, exposure, and saturation, then writes ranked, explainable candidates to `.video-agent/highlights.json`. The score is a deterministic visual heuristic, not an aesthetic judgment; every candidate remains review-only, and low-motion scenic footage is explicitly preserved as a possible highlight. The `analyze` command scans the folder and extracts media metadata using ffprobe. The `understand` command detects scenes, extracts representative frames, sends each frame to the local Ollama model, and writes `.video-agent/understanding.json`. The `transcribe` command uses faster-whisper locally to recognize speech from video audio streams and writes `.video-agent/transcription.json`. The `detect-silence` command runs FFmpeg’s `silencedetect` filter and writes `.video-agent/silence-analysis.json`, correlating quiet intervals with transcript timestamps when available. The `propose-edits` command combines those review candidates with optional scene descriptions and writes a human-reviewable `.video-agent/edit-proposal.json`. Use `review-edits` to approve/reject each item interactively, or pass repeated `--approve ITEM_ID` and `--reject ITEM_ID` options for scripted review. Skipped items remain pending. Once every item has an explicit decision, `build-timeline` writes `.video-agent/timeline.json` containing the ordered source intervals to retain. This validated timeline is an edit-decision artifact only. After every cut has been reviewed, `render` uses FFmpeg to create `.video-agent/rendered.mp4` as an H.264/AAC MP4. The renderer checks the timeline and source files, keeps all source media untouched, refuses to overwrite an existing output unless `--overwrite` is supplied, and probes the finished MP4 before reporting success. Clips with no audio receive generated silence so all segments can be concatenated consistently. The subtitles command reads the Whisper transcription and validated timeline, remaps speech captions through removed intervals and concatenated clips, and writes .video-agent/subtitles.srt. Render with --burn-subtitles to burn the SRT into the final output after concatenation; use --subtitle-font-size (12–72) and --subtitle-position (bottom, center, top) to adjust its appearance, or pass --subtitle-file for a custom SRT.

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

Inspect `.video-agent/footage-suggestions.json`, then explicitly approve candidate numbers per shot (candidate numbers are 1-based):

```bash
docker compose run --rm video-agent review-footage ./travel-test \
  --select shot-001=1 \
  --select shot-002=2
```

Repeat `--select SHOT_ID=CANDIDATE_NUMBER` for each shot. Use `--reject SHOT_ID` to clear an earlier assignment. Unselected shots remain unassigned. The project advances to `footage_mapped` only when every storyboard shot has an explicitly approved source interval. This stage does not approve the overall creative plan, alter source media, or render a video.


### Output aspect ratio and resolution

When a current `.video-agent/video-project-plan.json` exists, `render-draft` uses its requested aspect ratio and resolution (16:9, 9:16, or 1:1; 720p, 1080p, or 4K). The rendered dimensions are verified with ffprobe, and the render artifact tracks the project plan as an upstream dependency.

Use `--fit-mode crop` (default) to fill the requested frame with a centered crop, or `--fit-mode pad` to preserve the complete source frame with letterboxing/pillarboxing. Crop can remove content near the edges; pad preserves all source content. If no project plan exists, the first approved source clip continues to determine output dimensions.

```bash
docker compose run --rm video-agent render-draft ./travel-test --fit-mode pad
```
