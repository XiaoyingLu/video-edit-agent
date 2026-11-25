"""Deterministic cache helpers for validated rendered video outputs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def build_render_cache_key(dependencies: list[Path], config: dict) -> str:
    """Hash render settings and the complete contents of every input dependency."""
    digest = hashlib.sha256()
    digest.update(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())
    for path in sorted({item.expanduser().resolve() for item in dependencies}, key=str):
        if not path.is_file():
            raise FileNotFoundError(f"Render cache dependency is missing: {path}")
        digest.update(str(path).encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def restore_cached_render(cache_file: Path, destination: Path) -> bool:
    """Atomically copy a cached render into place; return False on a cache miss."""
    if not cache_file.is_file() or cache_file.stat().st_size == 0:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".video-agent-cache-",
            suffix=destination.suffix,
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        shutil.copy2(cache_file, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def store_render_cache(rendered_file: Path, cache_file: Path) -> None:
    """Atomically store a completed render under its deterministic cache key."""
    if not rendered_file.is_file() or rendered_file.stat().st_size == 0:
        raise FileNotFoundError(f"Cannot cache missing or empty render: {rendered_file}")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".video-agent-cache-",
            suffix=".mp4",
            dir=cache_file.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        shutil.copy2(rendered_file, temporary)
        os.replace(temporary, cache_file)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
