"""Deterministic cache helpers for validated rendered video outputs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def _load_fingerprint_index(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_fingerprint_index(path: Path, index: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".fingerprints-",
            suffix=".json",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(index, stream, sort_keys=True, separators=(",", ":"))
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_render_cache_key(
    dependencies: list[Path], config: dict, *, fingerprint_index_path: Path | None = None
) -> str:
    """Hash render settings and dependencies, reusing hashes while metadata is stable."""
    digest = hashlib.sha256()
    digest.update(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())
    index = _load_fingerprint_index(fingerprint_index_path)
    updated = False
    for path in sorted({item.expanduser().resolve() for item in dependencies}, key=str):
        if not path.is_file():
            raise FileNotFoundError(f"Render cache dependency is missing: {path}")
        stat = path.stat()
        metadata = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        cached = index.get(str(path))
        if (
            isinstance(cached, dict)
            and isinstance(cached.get("sha256"), str)
            and all(cached.get(key) == value for key, value in metadata.items())
        ):
            file_hash = cached.get("sha256")
        else:
            file_hash = _file_digest(path)
            index[str(path)] = {**metadata, "sha256": file_hash}
            updated = True
        digest.update(str(path).encode())
        digest.update(b"\\0")
        digest.update(str(file_hash).encode())
    if updated and fingerprint_index_path is not None:
        _write_fingerprint_index(fingerprint_index_path, index)
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
