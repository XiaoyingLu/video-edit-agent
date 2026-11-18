# Music-Aware Timeline Refinement Design

**Status:** Approved by the user on 2026-11-18

## Goal

Turn the existing review-only beat-alignment suggestions into an optional, non-destructive workflow that can apply user-accepted timing adjustments to adjacent shots while preserving valid source intervals and requiring a fresh timeline approval before rendering.

## Product decisions

- Keep the product CLI-first, local-first, and built around FFmpeg.
- Keep hybrid creative autonomy: the agent proposes; the user reviews and explicitly accepts or rejects each recommendation.
- Keep PR #34's `suggest-music-sync` behavior advisory and unchanged. Refinement is a separate follow-on capability.
- Do not force every cut onto a beat. Only eligible suggestions within the existing confidence/tolerance rules are candidates.
- Accepted recommendations may adjust the source in/out points of the two adjacent shots, but only within the actual source media duration and configured minimum usable shot duration.
- Preserve the original timeline. Applying accepted recommendations creates a separate timeline revision.
- A revised timeline is not render-approved. The user must review and approve the revised timeline before rendering.
- Keep HyperFrames optional and out of this feature.

## User workflow

```bash
video-agent suggest-music-sync ./travel-video
video-agent review-music-sync ./travel-video
video-agent review-timeline ./travel-video --timeline .video-agent/music-sync-revisions/revision-001.json
video-agent render-draft ./travel-video --timeline .video-agent/music-sync-revisions/revision-001.json
```

The review command presents each eligible transition with:
- adjacent clip IDs and transition type;
- current transition position and proposed beat/downbeat anchor;
- signed timing adjustment;
- which adjacent source interval(s) will change;
- resulting source in/out points and shot durations;
- feasibility result and a concise reason.

The user accepts or rejects each recommendation. Rejected and skipped recommendations do not alter the timeline. A command may support explicit `--accept` / `--reject` identifiers for non-interactive use, following existing CLI conventions.

## Refinement behavior

For each accepted recommendation:

1. Convert the suggested timeline-boundary shift into a source-time adjustment for the adjacent clips.
2. Select the adjacent clip that can safely give up or gain source duration; prefer preserving the outgoing clip when the incoming clip has sufficient available handles, otherwise use the outgoing clip when safe.
3. Verify the adjusted source start/end remain within the probed source duration.
4. Preserve each clip's minimum usable duration. Use a conservative default of 1.0 second, configurable only if the existing project conventions support it.
5. Preserve the timeline's transition type and transition duration. Recalculate total proposed duration from clip durations and overlap durations.
6. Validate the complete revised timeline with the existing Pydantic timeline model and any existing timeline invariants.
7. If a recommendation cannot be applied safely, mark it infeasible, explain why, and leave both adjacent clips unchanged.

Recommendations that touch the same clip must be evaluated against the progressively updated in-memory revision, not independently against the original. If a later recommendation conflicts with an earlier accepted adjustment, reject that later adjustment as infeasible rather than silently undoing an accepted change.

## Review state, revisions, and provenance

- Never overwrite `.video-agent/optimized-timeline.json`.
- Store each successfully applied revision under `.video-agent/music-sync-revisions/` using a monotonically increasing revision number.
- Persist the review record, including accepted/rejected/skipped/infeasible decisions, the source suggestion artifact, source timeline path, and resulting revision path.
- Register the revision artifact with dependencies on the source timeline and music-sync suggestions. Preserve the existing artifact-manifest dependency validation behavior.
- The new revision starts with `timeline_approved=false`. The existing `review-timeline` command is the explicit approval gate.
- Rendering must use the revision path explicitly. No render is started by `review-music-sync`.
- If the source timeline or music analysis changes, existing artifact validation must prevent applying stale suggestions.
- If a revised timeline changes after approval, the normal artifact dependency/staleness rules must prevent rendering stale output.

## Scope boundaries

Included:
- interactive and/or explicit accept/reject review of existing music-sync recommendations;
- safe adjacent source interval adjustment for hard cuts, crossfades, and dissolves;
- source duration probing and minimum shot-duration validation;
- conflict handling for adjacent recommendations;
- versioned timeline revisions and review provenance;
- tests and CLI documentation.

Excluded:
- automatic narrative rearrangement or clip reordering;
- music-section or phrase understanding;
- automatically aligning every cut;
- changing transition types or transition durations;
- changing source media;
- automatic rendering or automatic approval;
- HyperFrames integration or a new renderer.

## Validation strategy

Tests must cover:
- hard-cut boundary adjustment;
- crossfade/dissolve midpoint semantics;
- source-start and source-end bounds;
- insufficient source handles;
- minimum shot duration;
- two adjacent recommendations that compete for the same clip;
- invalid resulting timeline duration/overlap;
- stale suggestion or source timeline artifacts;
- revision numbering and preservation of the original timeline;
- persistence of accepted, rejected, skipped, and infeasible decisions;
- revised timeline remains unapproved until the existing timeline review gate;
- CLI review behavior and render path handoff.

## Architectural fit

Keep music analysis and recommendation logic in the music domain. Keep timeline schema validation and persistence consistent with the existing editing workflow. Keep the CLI as orchestration and presentation only. FFmpeg remains unaware of music-sync recommendation logic and receives only a reviewed, approved timeline.
