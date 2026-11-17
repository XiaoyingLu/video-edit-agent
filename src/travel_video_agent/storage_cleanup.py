"""Bounded cleanup for regenerable frame caches in a video project."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

CACHE_DIRECTORIES = (Path(".video-agent/frames"), Path(".video-agent/highlight-frames"))


@dataclass(frozen=True)
class CleanupReport:
    """Summary of a generated-cache cleanup operation."""

    scanned_files: int
    candidate_files: int
    removed_files: int
    reclaimed_bytes: int
    remaining_bytes: int
    dry_run: bool


def _referenced_frames(root: Path) -> set[Path]:
    """Return frame paths referenced by the current understanding artifact."""
    path = root / ".video-agent" / "understanding.json"
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()

    referenced: set[Path] = set()
    if not isinstance(payload, dict) or not isinstance(payload.get("scenes"), list):
        return set()
    for scene in payload["scenes"]:
        if not isinstance(scene, dict) or not isinstance(scene.get("observations"), list):
            continue
        for observation in scene["observations"]:
            if not isinstance(observation, dict):
                continue
            frame_path = observation.get("frame_path")
            if not isinstance(frame_path, str):
                continue
            candidate = (root / frame_path).resolve()
            try:
                candidate.relative_to((root / ".video-agent").resolve())
            except ValueError:
                continue
            referenced.add(candidate)
    return referenced


def cleanup_generated_storage(
    folder: Path,
    *,
    max_age_days: float = 7,
    max_cache_gb: float = 5,
    keep_artifacts: bool = False,
    dry_run: bool = False,
) -> CleanupReport:
    """Prune old/oversized extracted-frame caches without touching durable artifacts.

    Only regular files under the two allowlisted frame-cache directories are eligible.
    Model weights, JSON/SRT artifacts, manifests, renders, and source media are never
    candidates. When keep_artifacts is enabled, frames referenced by understanding.json
    are protected as well.
    """
    if not math.isfinite(max_age_days) or max_age_days < 0:
        raise ValueError("max_age_days must be zero or greater")
    if not math.isfinite(max_cache_gb) or max_cache_gb <= 0:
        raise ValueError("max_cache_gb must be greater than zero")

    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project folder does not exist: {root}")

    artifact_root = (root / ".video-agent").resolve()
    now = datetime.now(UTC)
    cutoff = now - timedelta(days=max_age_days)
    protected = _referenced_frames(root) if keep_artifacts else set()
    candidates: list[tuple[Path, int, float]] = []
    scanned_files = 0

    for relative_dir in CACHE_DIRECTORIES:
        cache_dir = root / relative_dir
        if not cache_dir.is_dir() or cache_dir.is_symlink():
            continue
        for path in cache_dir.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            resolved = path.resolve()
            try:
                resolved.relative_to(artifact_root)
            except ValueError:
                continue
            scanned_files += 1
            if resolved in protected:
                continue
            stat = path.stat()
            candidates.append((path, stat.st_size, stat.st_mtime))

    total_bytes = sum(size for _, size, _ in candidates)
    max_bytes = int(max_cache_gb * 1024**3)
    selected: list[Path] = []
    selected_bytes = 0

    # Age-based pruning is deterministic and takes priority over the size budget.
    for path, size, modified in sorted(candidates, key=lambda item: (item[2], item[0].as_posix())):
        modified_at = datetime.fromtimestamp(modified, tz=UTC)
        if modified_at < cutoff:
            selected.append(path)
            selected_bytes += size

    remaining = total_bytes - selected_bytes
    if remaining > max_bytes:
        selected_set = set(selected)
        for path, size, _ in sorted(
            candidates, key=lambda item: (item[2], item[0].as_posix())
        ):
            if path in selected_set:
                continue
            if remaining <= max_bytes:
                break
            selected.append(path)
            selected_set.add(path)
            selected_bytes += size
            remaining -= size

    removed_files = 0
    if not dry_run:
        for path in selected:
            try:
                path.unlink()
                removed_files += 1
            except FileNotFoundError:
                continue
        for relative_dir in CACHE_DIRECTORIES:
            cache_dir = root / relative_dir
            if cache_dir.is_dir() and not cache_dir.is_symlink():
                for directory in sorted(cache_dir.rglob("*"), reverse=True):
                    if directory.is_dir() and not directory.is_symlink():
                        try:
                            directory.rmdir()
                        except OSError:
                            pass
                try:
                    cache_dir.rmdir()
                except OSError:
                    pass
    return CleanupReport(
        scanned_files=scanned_files,
        candidate_files=len(selected),
        removed_files=removed_files,
        reclaimed_bytes=selected_bytes,
        remaining_bytes=max(0, total_bytes - selected_bytes),
        dry_run=dry_run,
    )
