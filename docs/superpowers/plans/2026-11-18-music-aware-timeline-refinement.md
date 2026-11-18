# Music-Aware Timeline Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CLI review workflow that applies explicitly accepted beat-alignment suggestions to a separate, safely validated timeline revision without modifying the original timeline or approving/rendering the revision.

**Architecture:** Keep beat suggestions in `music/sync.py` and add a focused `music/refinement.py` module for decision records, safe timeline adjustment, revision persistence, and artifact provenance. The CLI orchestrates interactive or flag-based review, while the existing `review-timeline` command remains the approval gate and FFmpeg renderer remains unchanged.

**Tech Stack:** Python, Pydantic, Typer, Rich, pytest, existing artifact tracking and FFprobe integration.

**Spec:** `docs/superpowers/specs/2026-11-18-music-aware-timeline-refinement-design.md`

## Global Constraints

- Keep PR #34's `suggest-music-sync` behavior advisory and unchanged.
- Never overwrite `.video-agent/optimized-timeline.json`.
- Store revisions under `.video-agent/music-sync-revisions/` with monotonically increasing revision numbers.
- A newly created revision must have `timeline_approved=false`.
- Never modify source media or invoke rendering from the review command.
- Source intervals must remain within probed source duration and each resulting clip must be at least 1.0 second.
- Preserve transition type and transition duration.
- Use existing artifact tracking and timeline Pydantic validation; do not add a renderer dependency.

## Review Focus

- A source file missing or unreadable during review must produce a clear error and no revision.
- A clip with insufficient source handles must be marked infeasible and remain unchanged.
- Recommendations sharing an adjacent clip must be applied sequentially against the evolving in-memory timeline; later conflicts must not undo earlier accepted changes.
- A stale suggestion or source timeline must be rejected before any revision is written.
- Re-running review must create a new revision and preserve prior revisions and review decisions.

---

### Task 1: Model refinement decisions and revision reports

**Files:**
- Create: `src/travel_video_agent/music/refinement.py`
- Test: `tests/test_music_sync_refinement.py`

**Interfaces:**
- Consumes: `MusicSyncSuggestions`, `ProjectDraftTimeline`, `DraftTimelineClip` from existing modules.
- Produces: `MusicSyncReviewDecision`, `MusicSyncReviewResult`, and `apply_music_sync_review(...)` used by later CLI integration.

- [ ] **Step 1: Write failing schema tests**
  - Add `test_review_decision_records_accepted_rejected_and_infeasible_states`: validate decisions `accepted`, `rejected`, `skipped`, and `infeasible`.
  - Add `test_review_result_records_source_and_revision_provenance`: assert source timeline, suggestion artifact, revision path, and per-boundary decisions are represented.
  - Add `test_review_decision_rejects_unknown_status`: assert Pydantic rejects an unsupported status.

- [ ] **Step 2: Run the focused tests and verify they fail**
  - Run: `pytest tests/test_music_sync_refinement.py -q`
  - Expected: FAIL because the refinement module and models do not exist.

- [ ] **Step 3: Implement the review models**
  - Define a strict Pydantic `MusicSyncReviewDecision` with `boundary_id`, `status: Literal["accepted", "rejected", "skipped", "infeasible"]`, and optional `reason`.
  - Define `MusicSyncReviewResult` with `schema_version`, `source_timeline_path`, `suggestions_path`, `revision_path`, `decisions`, and `review_required: Literal[True]`.
  - Define boundary IDs deterministically as `"{previous_clip_id}->{next_clip_id}"`.

- [ ] **Step 4: Run focused tests and verify they pass**
  - Run: `pytest tests/test_music_sync_refinement.py -q`
  - Expected: PASS.

- [ ] **Step 5: Commit**
  - Commit message: `feat: model music sync review decisions`.

### Task 2: Apply accepted timing changes safely

**Files:**
- Modify: `src/travel_video_agent/music/refinement.py`
- Test: `tests/test_music_sync_refinement.py`

**Interfaces:**
- Consumes: Task 1 models; `probe_video(path, project_root)` from `media.py`; existing timeline and suggestion schemas.
- Produces: `apply_music_sync_review(folder: Path, *, suggestions_path: Path | None = None, timeline_path: Path | None = None, accept_ids: set[str], reject_ids: set[str], minimum_clip_duration_seconds: float = 1.0) -> MusicSyncReviewResult`.

- [ ] **Step 1: Write failing adjustment tests**
  - Add `test_accepting_hard_cut_recommendation_moves_boundary_and_preserves_total_duration`.
  - Add `test_accepting_crossfade_recommendation_uses_transition_midpoint`.
  - Add `test_accepting_dissolve_recommendation_uses_transition_midpoint`.
  - Add `test_adjustment_never_exceeds_source_duration`.
  - Add `test_adjustment_preserves_minimum_clip_duration`.
  - Add `test_infeasible_adjustment_leaves_both_clips_unchanged`.
  - Add `test_conflicting_adjacent_recommendations_are_processed_sequentially`.
  - Add `test_revised_timeline_recalculates_proposed_duration_and_validates`.

- [ ] **Step 2: Run the focused tests and verify they fail**
  - Run: `pytest tests/test_music_sync_refinement.py -q`
  - Expected: FAIL because the application function is not implemented.

- [ ] **Step 3: Implement deterministic safe adjustment**
  - Load and validate the source timeline and suggestion report, confirming the report references the same timeline.
  - Probe each distinct source video once with `probe_video` and cache `duration_seconds`.
  - For each accepted boundary in timeline order, calculate the requested shift from `adjustment_seconds`; use the transition midpoint semantics already encoded in the suggestion report.
  - Apply the shift by extending/shortening the outgoing clip's source end and applying the inverse adjustment to the incoming clip's source start, preserving overall project duration. Reject the adjustment if either resulting source interval leaves its source bounds or either clip falls below 1.0 second.
  - Evaluate each later recommendation against the already adjusted in-memory clips. Record unsafe or conflicting recommendations as `infeasible` without changing the current in-memory timeline.
  - Recalculate `proposed_duration_seconds`, set `timeline_approved=False`, preserve decisions and all transition metadata, then validate with `ProjectDraftTimeline.model_validate`.
  - Keep rejected/skipped recommendations out of the timeline mutation path.

- [ ] **Step 4: Run focused tests and verify they pass**
  - Run: `pytest tests/test_music_sync_refinement.py -q`
  - Expected: PASS.

- [ ] **Step 5: Commit**
  - Commit message: `feat: safely apply accepted music sync adjustments`.

### Task 3: Persist versioned revisions and review provenance

**Files:**
- Modify: `src/travel_video_agent/music/refinement.py`
- Modify: `tests/test_music_sync_refinement.py`

**Interfaces:**
- Consumes: Task 2's `apply_music_sync_review`; `register_artifact` and `validate_artifact`.
- Produces: persisted revision JSON and review JSON under the project artifact directory.

- [ ] **Step 1: Write failing persistence and safety tests**
  - Add `test_review_creates_revision_001_without_overwriting_original_timeline`.
  - Add `test_second_review_creates_revision_002_and_preserves_revision_001`.
  - Add `test_review_record_persists_accepted_rejected_skipped_and_infeasible_decisions`.
  - Add `test_revision_is_registered_with_timeline_and_suggestion_dependencies`.
  - Add `test_stale_suggestions_are_rejected_without_creating_revision`.
  - Add `test_missing_source_video_fails_without_creating_revision`.

- [ ] **Step 2: Run the focused tests and verify they fail**
  - Run: `pytest tests/test_music_sync_refinement.py -q`
  - Expected: FAIL because persistence is not implemented.

- [ ] **Step 3: Implement atomic versioned persistence**
  - Allocate the next `revision-NNN.json` path by scanning the revision directory; never overwrite an existing revision.
  - Write the validated timeline and review record via temporary files followed by atomic replacement.
  - Register the revision artifact with dependencies on the source timeline and suggestion report.
  - Validate both input artifacts before writing; if validation fails, leave the revision directory and source timeline unchanged.
  - Persist every recommendation's final review status and reason.

- [ ] **Step 4: Run focused tests and verify they pass**
  - Run: `pytest tests/test_music_sync_refinement.py -q`
  - Expected: PASS.

- [ ] **Step 5: Commit**
  - Commit message: `feat: persist versioned music sync revisions`.

### Task 4: Add the review-music-sync CLI workflow

**Files:**
- Modify: `src/travel_video_agent/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_music_sync_refinement.py`

**Interfaces:**
- Consumes: `apply_music_sync_review`, existing `MusicSyncSuggestions`, Typer and Rich.
- Produces: `video-agent review-music-sync PROJECT` with optional repeatable `--accept BOUNDARY_ID`, `--reject BOUNDARY_ID`, `--timeline PATH`, `--suggestions PATH`, and `--minimum-clip-duration` options.

- [ ] **Step 1: Write failing CLI tests**
  - Add `test_review_music_sync_command_help`: assert all options are shown.
  - Add `test_review_music_sync_command_accepts_and_persists_selected_boundary`: monkeypatch the review function and assert the accepted ID is passed.
  - Add `test_review_music_sync_command_rejects_overlapping_accept_and_reject_ids`: assert a non-zero exit and no review function call.
  - Add `test_review_music_sync_command_interactively_prompts_for_each_pending_boundary`: simulate accept/reject/skip input and assert the decisions passed to the application function.

- [ ] **Step 2: Run the focused CLI tests and verify they fail**
  - Run: `pytest tests/test_cli.py -q -k music_sync`
  - Expected: FAIL because the command does not exist.

- [ ] **Step 3: Implement CLI orchestration**
  - Add `review-music-sync` adjacent to `suggest-music-sync`.
  - If neither `--accept` nor `--reject` is supplied, display each recommendation with current/proposed boundary, transition type, signed delta, and the adjacent source intervals, then prompt for accept/reject/skip.
  - If explicit flags are supplied, treat unlisted recommendations as skipped.
  - Reject IDs that do not exist and IDs present in both accept and reject sets.
  - Call `apply_music_sync_review`, print per-boundary outcomes and the created revision path, and explicitly state that the revision still requires `review-timeline` approval and no rendering occurred.
  - Catch expected file, validation, and media-probe errors and exit non-zero without claiming a revision was created.

- [ ] **Step 4: Run focused CLI tests and verify they pass**
  - Run: `pytest tests/test_cli.py -q -k music_sync`
  - Expected: PASS.

- [ ] **Step 5: Commit**
  - Commit message: `feat: add interactive music sync review command`.

### Task 5: Document the workflow and verify integration

**Files:**
- Modify: `README.md`
- Test: existing music-sync, timeline-review, CLI, and full test suites.

- [ ] **Step 1: Update README**
  - Document the review command, versioned revision location, explicit `review-timeline --timeline` approval step, and explicit `render-draft --timeline` usage.
  - State that source media and the original optimized timeline remain unchanged.

- [ ] **Step 2: Run targeted regression tests**
  - Run: `pytest tests/test_music_sync.py tests/test_music_sync_refinement.py tests/test_cli.py -q`
  - Expected: PASS.

- [ ] **Step 3: Run full verification**
  - Run: `pytest -q`
  - Run: `ruff check .`
  - Run: `ruff format --check .`
  - Expected: all tests pass and Ruff reports no issues.

- [ ] **Step 4: Review the complete diff**
  - Verify PR #34's existing suggestion behavior is unchanged.
  - Verify no source media, original timeline, renderer behavior, or HyperFrames integration is modified.
  - Verify every created revision is unapproved and artifact dependencies are recorded.

- [ ] **Step 5: Commit**
  - Commit message: `docs: explain music sync timeline refinement workflow`.
