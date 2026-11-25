# Reusable templates

Reusable templates are versioned YAML data files. They describe styling, title
slots, transitions, music mix settings, and safe render defaults. They are not
executable recipes: unknown fields are rejected, and animation/transition
values are constrained to built-in renderer options. Raw FFmpeg expressions
are never accepted.

## Template locations

Templates are resolved in this order:

1. `.video-agent/templates/<id>.yaml` in the selected project
2. `~/.video-agent/templates/<id>.yaml` in the user's home directory
3. The templates shipped in `video_edit_agent/template_library`

A project template can therefore override a home or built-in template with
the same ID. IDs use lowercase letters, digits, and hyphens.

## CLI discovery and validation

List available templates and inspect their resolved source:

```bash
video-agent template list
video-agent template list --project ./my-project
video-agent template show travel-montage
video-agent template show cinematic --project ./my-project
video-agent template validate travel-montage
```

`show` prints the fully resolved data, including inherited settings.
`validate` checks the selected template, its inheritance chain, schema, and
asset paths without writing to the project. Both commands honor the same
project-over-home-over-built-in precedence as rendering workflows.

Apply a template directly through the template command:

```bash
video-agent template apply ./my-project travel-montage
```

Optional `--timeline`, `--plan`, and `--output` arguments select the input
and output files. The command refuses to overwrite an existing proposal.

## Preview and compare templates

Render two or three opening-card designs over the same frame from the first
approved timeline clip:

```bash
video-agent template preview ./my-project cinematic vlog travel-montage
```

The command samples the midpoint of the first approved source clip, uses the
story treatment's working title and logline, and renders one JPEG per template.
It also writes `.video-agent/template-previews/index.html` for a side-by-side
comparison. Use `--output-dir` to choose another directory inside the project.
Existing preview files are never overwritten. Previewing does not create or
approve a title-overlay artifact and does not modify source media.

## Schema

Each file declares `id` and `version`. Optional `extends` inherits the
parent's values; mappings are deep-merged and lists are replaced. Inheritance
cycles and filename/ID mismatches are rejected.

Supported sections are:

- `typography`: font family/filename and hex text/accent colors
- `title_styles`: placement, relative size, box opacity, built-in animation,
  and optional accent bar
- `sequences`: named sequences of content slots and title style keys
- `transitions`: built-in transition choices for default, scene changes, and beats
- `music`: mode, volume, fade durations, and ducking
- `render`: crop/pad fit and opt-in beat punch-in
- `subtitles`: built-in subtitle style (`classic`, `cinematic`, `boxed`, or `minimal`), position, and font size
- `tags`: descriptive mood and use-case labels
- `assets`: optional paths relative to the template file's directory

Supported content slots are `{title}`, `{logline}`, `{closing_line}`,
`{date}`, and `{location}`. Asset paths must remain inside the directory
containing the template file, including after symlink resolution.

## Built-ins

The initial library includes `cinematic`, `vlog`, `travel-montage`, and
`social`. The existing render preset workflow remains unchanged while the
template library is introduced in stages.

## Applying a template to title sequences

After approving the timeline and creating a story treatment, generate a
review-only opening/closing title proposal with:

```bash
video-agent suggest-title-sequences ./my-project --template travel-montage
```

The command reads the template's `sequences.opening` and
`sequences.closing` slots, fills them from the story treatment, and writes
`.video-agent/title-overlays.json`. The artifact records the template ID,
version, and source file, and tracks that file as a dependency so a changed
template makes the proposal stale.

Generated overlays always start with `decision: pending`. Applying a
template never approves titles or bypasses `review-title-overlays`.
Existing schema 1.0 title-overlay documents remain readable; newly generated
template-driven documents use schema 1.1.

## Save a reusable template from a project

Capture a project's approved title styling and transition choices, plus the
render profile associated with its creative style:

```bash
video-agent template save ./my-project my-travel-look --from-project
```

The default destination is `~/.video-agent/templates/my-travel-look.yaml`,
so the result can be reused across projects. Use `--output` to save it in a
project-local template directory or another location. Existing files are
never overwritten.

Only approved opening/closing text-overlay styling is captured. Actual title
text, footage paths, assets, clip selections, timings, and approval decisions
are not copied. Transition defaults are inferred from the approved timeline.
The project's creative style selects the starting render/music profile;
because past `render-draft` CLI overrides are not persisted as project
settings, use `--music-mode`, `--music-volume`, `--music-fade-in`,
`--music-fade-out`, and `--music-ducking/--no-music-ducking` to record
custom mix values explicitly. The command prints notes when title styling
must fall back to safe defaults.

## Tag-based recommendations

Ask for up to three explainable matches using the tags declared by templates:

```bash
video-agent template recommend --mood warm --best-for travel
video-agent template recommend --mood energetic --best-for short-form --limit 2
```

Repeat `--mood` or `--best-for` to combine multiple tags. Matching is exact
after case, whitespace, underscore, and hyphen normalization; it does not use
embeddings or make semantic guesses. Each matching `best_for` tag contributes
two points and each matching `mood` tag contributes one point. Results are
ranked by score, with template ID as a deterministic tie-breaker, and show the
matched tags and source. No match is also a valid outcome. Recommendations
never apply a template or alter project files. Use `--project` to include
project-local templates and preserve the usual project-over-home-over-built-in
lookup precedence.

## Template-driven subtitle styling

Use a reusable template to supply default subtitle styling when burning captions into a render:

```bash
video-agent render-video ./my-project --burn-subtitles --template cinematic
```

The template's `subtitles` section supports only built-in styles, top/center/bottom positioning, and a font size from 12 to 72. Individual CLI options such as `--subtitle-style`, `--subtitle-position`, and `--subtitle-font-size` override the template values. Without `--template`, existing render defaults remain unchanged. Templates never contain raw ASS or FFmpeg expressions.
