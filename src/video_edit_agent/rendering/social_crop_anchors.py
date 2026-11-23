"""Editable, timeline-bound manual crop anchors for social exports."""

import hashlib
import json
from pathlib import Path
from typing import Any

from ..atomic_io import write_json_atomic
from ..editing.timeline_schemas import ProjectDraftTimeline


def _timeline_digest(timeline_file: Path) -> str:
    return hashlib.sha256(timeline_file.read_bytes()).hexdigest()


def _resolve_project_file(root: Path, path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = root / expanded
    resolved = expanded.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the project folder") from exc
    return resolved


def create_crop_anchor_template(
    folder: Path,
    *,
    timeline_path: Path | None = None,
    output_path: Path | None = None,
) -> tuple[Path, list[str]]:
    """Create a template tied to the exact current timeline bytes."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project folder does not exist: {root}")
    timeline_file = _resolve_project_file(
        root,
        timeline_path or root / ".video-agent" / "optimized-timeline.json",
        "Timeline path",
    )
    if not timeline_file.is_file():
        raise FileNotFoundError(f"Missing optimized timeline: {timeline_file}")
    timeline = ProjectDraftTimeline.model_validate_json(timeline_file.read_text(encoding="utf-8"))
    if Path(timeline.project_path).resolve() != root:
        raise ValueError("optimized timeline belongs to a different project folder")
    if not timeline.timeline_approved:
        raise ValueError("optimized timeline must be fully approved before creating crop anchors")
    clip_ids = [clip.clip_id for clip in timeline.clips]
    destination = _resolve_project_file(
        root,
        output_path or root / ".video-agent" / "social-crop-anchors.json",
        "Crop-anchor output path",
    )
    if destination.exists():
        raise FileExistsError(f"Crop-anchor file already exists: {destination}")
    document = {
        "schema_version": "1.0",
        "project_path": str(root),
        "source_timeline": timeline_file.relative_to(root).as_posix(),
        "timeline_sha256": _timeline_digest(timeline_file),
        "available_clip_ids": clip_ids,
        "anchors": [],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(destination, document)
    return destination, clip_ids


def save_crop_anchors(
    folder: Path,
    *,
    timeline_path: Path | None = None,
    anchors: list[dict[str, Any]],
    output_path: Path | None = None,
) -> Path:
    """Validate and atomically save manual anchors for an approved timeline."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project folder does not exist: {root}")
    timeline_file = _resolve_project_file(
        root,
        timeline_path or root / ".video-agent" / "optimized-timeline.json",
        "Timeline path",
    )
    if not timeline_file.is_file():
        raise FileNotFoundError(f"Missing optimized timeline: {timeline_file}")
    timeline = ProjectDraftTimeline.model_validate_json(timeline_file.read_text(encoding="utf-8"))
    if Path(timeline.project_path).resolve() != root:
        raise ValueError("optimized timeline belongs to a different project folder")
    if not timeline.timeline_approved:
        raise ValueError("approve the complete timeline before saving crop anchors")
    clip_ids = [clip.clip_id for clip in timeline.clips]
    if not isinstance(anchors, list):
        raise ValueError("crop-anchor anchors must be a JSON array")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in anchors:
        if not isinstance(entry, dict) or set(entry) != {"clip_id", "x", "y"}:
            raise ValueError("each crop anchor must contain only clip_id, x, and y")
        clip_id = entry["clip_id"]
        if not isinstance(clip_id, str) or clip_id not in clip_ids:
            raise ValueError(f"unknown crop-anchor clip_id: {clip_id}")
        if clip_id in seen:
            raise ValueError(f"duplicate crop-anchor clip_id: {clip_id}")
        seen.add(clip_id)
        coordinates: dict[str, float] = {}
        for axis in ("x", "y"):
            value = entry[axis]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"crop-anchor {axis} must be a number between 0 and 1")
            if not 0 <= value <= 1:
                raise ValueError(f"crop-anchor {axis} must be between 0 and 1")
            coordinates[axis] = float(value)
        validated.append({"clip_id": clip_id, **coordinates})
    destination = _resolve_project_file(
        root,
        output_path or root / ".video-agent" / "social-crop-anchors.json",
        "Crop-anchor output path",
    )
    document = {
        "schema_version": "1.0",
        "project_path": str(root),
        "source_timeline": timeline_file.relative_to(root).as_posix(),
        "timeline_sha256": _timeline_digest(timeline_file),
        "available_clip_ids": clip_ids,
        "anchors": validated,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(destination, document)
    return destination


def load_crop_anchors(
    root: Path,
    timeline_file: Path,
    timeline: ProjectDraftTimeline,
    anchor_file: Path,
) -> dict[str, tuple[float, float]]:
    """Load and validate manual anchors against the exact timeline and project."""
    root = root.resolve()
    source = _resolve_project_file(root, anchor_file, "Crop-anchor file")
    if not source.is_file():
        raise FileNotFoundError(f"Crop-anchor file does not exist: {source}")
    try:
        document: Any = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid crop-anchor JSON: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != "1.0":
        raise ValueError("unsupported crop-anchor schema_version")
    if document.get("project_path") != str(root):
        raise ValueError("crop anchors belong to a different project folder")
    expected_timeline = timeline_file.resolve().relative_to(root).as_posix()
    if document.get("source_timeline") != expected_timeline:
        raise ValueError("crop anchors reference a different optimized timeline")
    if document.get("timeline_sha256") != _timeline_digest(timeline_file):
        raise ValueError("crop-anchor template is stale; regenerate it for the current timeline")
    current_ids = [clip.clip_id for clip in timeline.clips]
    if document.get("available_clip_ids") != current_ids:
        raise ValueError("crop-anchor template clip IDs do not match the current timeline")
    entries = document.get("anchors")
    if not isinstance(entries, list):
        raise ValueError("crop-anchor anchors must be a JSON array")
    result: dict[str, tuple[float, float]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"clip_id", "x", "y"}:
            raise ValueError("each crop anchor must contain only clip_id, x, and y")
        clip_id = entry["clip_id"]
        if not isinstance(clip_id, str) or not clip_id:
            raise ValueError("crop-anchor clip_id must be a non-empty string")
        if clip_id not in current_ids:
            raise ValueError(f"unknown crop-anchor clip_id: {clip_id}")
        if clip_id in result:
            raise ValueError(f"duplicate crop-anchor clip_id: {clip_id}")
        coordinates = []
        for axis in ("x", "y"):
            value = entry[axis]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"crop-anchor {axis} must be a number between 0 and 1")
            if not 0 <= value <= 1:
                raise ValueError(f"crop-anchor {axis} must be between 0 and 1")
            coordinates.append(float(value))
        result[clip_id] = (coordinates[0], coordinates[1])
    return result
