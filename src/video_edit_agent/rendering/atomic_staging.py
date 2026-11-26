"""Per-run staging for generated media with checksum-gated atomic promotion."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from collections.abc import Callable


def sha256_file(path: Path) -> str:
    """Return the full SHA-256 digest of a staged file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunWorkspace:
    """Isolated staging area whose contents are removed unless explicitly promoted."""

    def __init__(self, project_root: Path):
        self.root = project_root.expanduser().resolve() / ".video-agent" / "workspaces"
        self.path: Path | None = None
        self.checksums: dict[str, str] = {}

    def __enter__(self) -> RunWorkspace:
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix="run-", dir=self.root))
        return self

    def stage_path(self, name: str) -> Path:
        if self.path is None:
            raise RuntimeError("run workspace has not been entered")
        candidate = (self.path / name).resolve()
        if not candidate.is_relative_to(self.path):
            raise ValueError("staged artifact name must stay inside the run workspace")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def checksum(self, staged: Path) -> str:
        if self.path is None or not staged.is_file() or not staged.is_relative_to(self.path):
            raise FileNotFoundError(f"staged artifact is missing or outside workspace: {staged}")
        digest = sha256_file(staged)
        self.checksums[staged.relative_to(self.path).as_posix()] = digest
        manifest = self.path / "checksums.json"
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.checksums, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, manifest)
        return digest

    def promote(
        self,
        staged: Path,
        destination: Path,
        *,
        finalize: Callable[[], None] | None = None,
    ) -> str:
        """Verify, atomically replace the active file, and roll back if finalization fails."""
        expected = None
        if self.path is not None and staged.is_relative_to(self.path):
            expected = self.checksums.get(staged.relative_to(self.path).as_posix())
        if expected is None or sha256_file(staged) != expected:
            raise ValueError(f"staged artifact checksum mismatch: {staged}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        backup = self.stage_path("previous-output.backup")
        had_previous = destination.is_file()
        if had_previous:
            shutil.copy2(destination, backup)
        try:
            os.replace(staged, destination)
            if sha256_file(destination) != expected:
                raise ValueError(f"promoted artifact checksum mismatch: {destination}")
            if finalize is not None:
                finalize()
        except Exception:
            destination.unlink(missing_ok=True)
            if had_previous and backup.is_file():
                os.replace(backup, destination)
            raise
        return expected

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.path is not None:
            shutil.rmtree(self.path, ignore_errors=True)
            self.path = None
