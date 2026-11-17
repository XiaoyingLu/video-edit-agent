"""Content-addressed provenance and dependency checks for project artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .media import discover_videos

MANIFEST_NAME = "artifact-manifest.json"
MANIFEST_VERSION = 1


class StaleArtifactError(ValueError):
    """Raised when a generated artifact no longer matches its inputs."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(root: Path, path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


def _manifest_path(root: Path) -> Path:
    return root / ".video-agent" / MANIFEST_NAME


def _load_manifest(root: Path) -> dict[str, Any]:
    path = _manifest_path(root)
    if not path.is_file():
        return {"version": MANIFEST_VERSION, "artifacts": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaleArtifactError(f"Could not read artifact manifest {path}: {exc}") from exc
    if payload.get("version") != MANIFEST_VERSION or not isinstance(
        payload.get("artifacts"), dict
    ):
        raise StaleArtifactError(
            f"Unsupported or invalid artifact manifest {path}; "
            "move it aside and regenerate project artifacts."
        )
    return payload


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    path = _manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _source_snapshot(root: Path, previous: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Fingerprint source videos, rehashing only when size or mtime changes."""
    previous = previous or {}
    snapshot: dict[str, dict[str, Any]] = {}
    artifact_dir = root / ".video-agent"
    sources = [
        path
        for path in discover_videos(root)
        if not path.is_relative_to(artifact_dir)
    ]
    for source in sorted(sources, key=lambda item: item.as_posix()):
        key = _key(root, source)
        stat = source.stat()
        old = previous.get(key, {})
        unchanged_stat = old.get("size") == stat.st_size and old.get("mtime_ns") == stat.st_mtime_ns
        digest = old.get("sha256") if unchanged_stat else _sha256(source)
        snapshot[key] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest or _sha256(source),
        }
    return snapshot


def register_artifact(
    folder: Path,
    artifact_path: Path,
    *,
    dependencies: list[Path | str] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Record an artifact's source snapshot, direct dependencies, and content hash.

    Passing dependencies=None when updating an existing artifact preserves its
    dependency edges (useful when a human review changes only the timeline JSON).
    """
    root = folder.expanduser().resolve()
    artifact = artifact_path.expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"Cannot register missing artifact: {artifact}")
    manifest = _load_manifest(root)
    artifacts = manifest["artifacts"]
    key = _key(root, artifact)
    existing = artifacts.get(key, {})
    shared_sources = existing.get("sources") or next(
        (item.get("sources") for item in artifacts.values() if item.get("sources")), {}
    )
    if dependencies is None:
        dependency_keys = list(existing.get("dependencies", []))
    else:
        dependency_keys = sorted(
            {
                _key(root, item if isinstance(item, Path) else root / item)
                for item in dependencies
            }
        )
    dependency_hashes: dict[str, str] = {}
    for dependency in dependency_keys:
        dependency_path = Path(dependency)
        if not dependency_path.is_absolute():
            dependency_path = root / dependency_path
        if not dependency_path.is_file():
            raise FileNotFoundError(f"Artifact dependency is missing: {dependency_path}")
        dependency_hashes[dependency] = _sha256(dependency_path)

    artifacts[key] = {
        "sha256": _sha256(artifact),
        "sources": _source_snapshot(root, shared_sources),
        "dependencies": dependency_keys,
        "dependency_hashes": dependency_hashes,
        "config": config if config is not None else existing.get("config", {}),
        "config_hash": hashlib.sha256(
            json.dumps(config if config is not None else existing.get("config", {}), sort_keys=True).encode()
        ).hexdigest(),
    }
    _write_manifest(root, manifest)


def validate_artifact(
    folder: Path,
    artifact_path: Path,
    *,
    expected_config: dict[str, Any] | None = None,
    _visited: set[str] | None = None,
) -> bool:
    """Validate a tracked artifact recursively; return False for legacy untracked files."""
    root = folder.expanduser().resolve()
    artifact = artifact_path.expanduser().resolve()
    manifest = _load_manifest(root)
    key = _key(root, artifact)
    record = manifest["artifacts"].get(key)
    if record is None:
        return False
    if not artifact.is_file():
        raise StaleArtifactError(f"Tracked artifact is missing: {key}; regenerate it.")

    visited = _visited if _visited is not None else set()
    if key in visited:
        raise StaleArtifactError(f"Artifact dependency cycle detected at {key}.")
    visited.add(key)
    try:
        if _sha256(artifact) != record.get("sha256"):
            raise StaleArtifactError(
                f"Artifact '{key}' was modified after it was generated; regenerate it."
            )

        current_sources = _source_snapshot(root, record.get("sources"))
        expected_sources = record.get("sources", {})
        if set(current_sources) != set(expected_sources):
            raise StaleArtifactError(
                f"Source video set changed since '{key}' was generated; "
                "rerun the affected analysis stages."
            )
        changed = [
            source
            for source, fingerprint in current_sources.items()
            if fingerprint["sha256"] != expected_sources[source].get("sha256")
        ]
        if changed:
            raise StaleArtifactError(
                f"Source video(s) changed since '{key}' was generated: "
                f"{', '.join(changed)}. Rerun the affected analysis stages."
            )

        for dependency in record.get("dependencies", []):
            dependency_path = Path(dependency)
            if not dependency_path.is_absolute():
                dependency_path = root / dependency_path
            if not dependency_path.is_file():
                raise StaleArtifactError(
                    f"Dependency '{dependency}' required by '{key}' is missing; "
                    "rerun the upstream stage."
                )
            validate_artifact(root, dependency_path, _visited=visited)
            actual_hash = _sha256(dependency_path)
            expected_hash = record.get("dependency_hashes", {}).get(dependency)
            if expected_hash and actual_hash != expected_hash:
                raise StaleArtifactError(
                    f"Dependency '{dependency}' changed after '{key}' was generated; "
                    "rerun the downstream stage."
                )

        if expected_config is not None:
            expected_hash = hashlib.sha256(
                json.dumps(expected_config, sort_keys=True).encode()
            ).hexdigest()
            if expected_hash != record.get("config_hash"):
                raise StaleArtifactError(
                    f"Configuration changed since '{key}' was generated; regenerate it."
                )
    finally:
        visited.remove(key)
    return True


def validate_all_artifacts(folder: Path) -> list[str]:
    """Return tracked artifacts that fail validation, without stopping at the first."""
    root = folder.expanduser().resolve()
    manifest = _load_manifest(root)
    stale: list[str] = []
    for key in sorted(manifest["artifacts"]):
        path = Path(key)
        if not path.is_absolute():
            path = root / path
        try:
            validate_artifact(root, path)
        except StaleArtifactError as exc:
            stale.append(f"{key}: {exc}")
    return stale


def record_artifact_dependencies(
    folder: Path,
    artifact_path: Path,
    dependencies: list[str],
    *,
    config: dict[str, Any] | None = None,
) -> None:
    """Convenience wrapper for artifacts stored beneath the project root."""
    register_artifact(folder, artifact_path, dependencies=dependencies, config=config)
