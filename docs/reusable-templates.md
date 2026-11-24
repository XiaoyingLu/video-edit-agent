# Reusable templates

Reusable templates are versioned YAML data files. They describe styling, title
slots, transitions, music mix settings, and safe render defaults. They are not
executable recipes: unknown fields are rejected, and animation/transition
values are constrained to built-in renderer options. Raw FFmpeg expressions
are never accepted.

## Template locations

Templates are resolved in this order:

1. `.video-agent/templates/<id>.yaml` in the current project
2. `~/.video-agent/templates/<id>.yaml` in the user's home directory
3. The templates shipped in `video_edit_agent/template_library`

A project template can therefore override a home or built-in template with
the same ID. IDs use lowercase letters, digits, and hyphens.

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
