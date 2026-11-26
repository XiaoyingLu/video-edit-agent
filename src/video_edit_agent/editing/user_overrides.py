"""User-owned edit overrides kept separate from agent-generated artifacts.

Overrides are intentionally excluded from the artifact checksum graph. They are
keyed by stable clip IDs and layered onto newly generated timelines after their
upstream artifacts have passed dependency validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..atomic_io import write_json_atomic

OVERRIDE_MANIFEST_NAME = "user-overrides.json"
OVERRIDE_MANIFEST_VERSION = 1
_ALLOWED_FIELDS = {
    "source_start_seconds",
    "source_end_seconds",
    "caption",
    "gain_db",
    "sequence_index",
    "transition_in",
    "transition_duration_seconds",
}


class UserOverrideError(ValueError):
    """Raised when a saved user edit cannot safely be applied to a new artifact."""


def _manifest_path(folder: Path) -> Path:
    return folder.expanduser().resolve() / ".video-agent" / OVERRIDE_MANIFEST_NAME


def load_user_overrides(folder: Path) -> dict[str, dict[str, Any]]:
    """Return the project override map, or an empty map for new projects."""
    path = _manifest_path(folder)
    if not path.is_file():
        return {}
    import json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UserOverrideError(f"Could not read user override manifest {path}: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("version") != OVERRIDE_MANIFEST_VERSION
        or not isinstance(payload.get("clips"), dict)
    ):
        raise UserOverrideError(f"Unsupported or invalid user override manifest: {path}")
    result: dict[str, dict[str, Any]] = {}
    for clip_id, values in payload["clips"].items():
        if not isinstance(clip_id, str) or not clip_id or not isinstance(values, dict):
            raise UserOverrideError("User override entries must be objects keyed by clip ID")
        unknown = set(values) - _ALLOWED_FIELDS
        if unknown:
            raise UserOverrideError(
                f"Unsupported override field(s) for clip '{clip_id}': {', '.join(sorted(unknown))}"
            )
        result[clip_id] = values
    return result


def save_user_overrides(
    folder: Path,
    overrides: dict[str, dict[str, Any]],
    *,
    replace: bool = False,
) -> Path:
    """Merge clip-level overrides into the separate user-owned manifest."""
    root = folder.expanduser().resolve()
    current = {} if replace else load_user_overrides(root)
    for clip_id, values in overrides.items():
        if not clip_id or not isinstance(values, dict):
            raise UserOverrideError("Overrides must be objects keyed by non-empty clip IDs")
        unknown = set(values) - _ALLOWED_FIELDS
        if unknown:
            raise UserOverrideError(
                f"Unsupported override field(s) for clip '{clip_id}': {', '.join(sorted(unknown))}"
            )
        current.setdefault(clip_id, {}).update(values)
    destination = _manifest_path(root)
    write_json_atomic(
        destination,
        {"version": OVERRIDE_MANIFEST_VERSION, "clips": current},
    )
    return destination


def apply_user_overrides(timeline):
    """Layer saved user edits onto a newly generated ProjectDraftTimeline."""
    overrides = load_user_overrides(Path(timeline.project_path))
    if not overrides:
        return timeline

    clips_by_id = {clip.clip_id: clip for clip in timeline.clips}
    unknown_ids = set(overrides) - set(clips_by_id)
    if unknown_ids:
        raise UserOverrideError(
            "Saved edits refer to clips no longer present in the regenerated timeline: "
            + ", ".join(sorted(unknown_ids))
            + ". Review or clear those entries in .video-agent/user-overrides.json."
        )

    patched = []
    changed = False
    for clip in timeline.clips:
        values = overrides.get(clip.clip_id, {})
        updates = dict(values)
        if any(getattr(clip, key) != value for key, value in values.items()):
            changed = True
        start = updates.get("source_start_seconds", clip.source_start_seconds)
        end = updates.get("source_end_seconds", clip.source_end_seconds)
        if start < clip.source_start_seconds - 0.001 or end > clip.source_end_seconds + 0.001:
            raise UserOverrideError(
                f"Saved trim for clip '{clip.clip_id}' no longer fits its regenerated source "
                "interval; review the override before continuing."
            )
        if end <= start:
            raise UserOverrideError(f"Saved trim for clip '{clip.clip_id}' has an invalid interval")
        if "source_start_seconds" in updates or "source_end_seconds" in updates:
            updates["source_start_seconds"] = round(start, 3)
            updates["source_end_seconds"] = round(end, 3)
            updates["duration_seconds"] = round(end - start, 3)
        updates.pop("sequence_index", None)
        try:
            patched.append(type(clip).model_validate({**clip.model_dump(), **updates}))
        except ValueError as exc:
            raise UserOverrideError(
                f"Invalid saved override for clip '{clip.clip_id}': {exc}"
            ) from exc

    # Preserve explicit user ordering where present; otherwise retain agent order.
    explicit_order = {
        clip_id: values["sequence_index"]
        for clip_id, values in overrides.items()
        if "sequence_index" in values
    }
    if explicit_order:
        original_positions = {clip.clip_id: index for index, clip in enumerate(patched)}
        patched.sort(
            key=lambda clip: (
                explicit_order.get(clip.clip_id, clip.sequence_index),
                original_positions[clip.clip_id],
            )
        )
        if [clip.clip_id for clip in patched] != [clip.clip_id for clip in timeline.clips]:
            changed = True

    for index, clip in enumerate(patched, start=1):
        clip.sequence_index = index
        if index == 1:
            clip.transition_in = "cut"
            clip.transition_duration_seconds = 0
        else:
            previous = patched[index - 2]
            if clip.transition_in != "cut":
                limit = min(clip.duration_seconds * 0.25, previous.duration_seconds * 0.25)
                clip.transition_duration_seconds = round(
                    min(clip.transition_duration_seconds, limit), 3
                )
                if clip.transition_duration_seconds <= 0:
                    clip.transition_in = "cut"
                    clip.transition_duration_seconds = 0

    proposed_duration = round(
        sum(clip.duration_seconds for clip in patched)
        - sum(clip.transition_duration_seconds for clip in patched[1:]),
        3,
    )
    if proposed_duration > timeline.target_duration_seconds + 0.02:
        raise UserOverrideError(
            "Reapplied user edits exceed the regenerated timeline's target duration."
        )
    return timeline.model_copy(
        update={
            "clips": patched,
            "proposed_duration_seconds": proposed_duration,
            "approved_duration_seconds": 0 if changed else timeline.approved_duration_seconds,
            "timeline_approved": False if changed else timeline.timeline_approved,
        },
        deep=True,
    )
