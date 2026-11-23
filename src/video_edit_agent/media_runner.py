"""Shared subprocess execution for FFmpeg and ffprobe operations."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from typing import Any


class MediaError(RuntimeError):
    """Raised when a media command cannot be started or completes unsuccessfully."""


def run_ffmpeg(
    cmd: Sequence[str],
    timeout: float,
    label: str,
    *,
    text: bool = True,
    error_type: type[MediaError] = MediaError,
) -> subprocess.CompletedProcess[Any]:
    """Run an FFmpeg/ffprobe command with consistent timeout and error handling.

    Set text=False to preserve binary stdout for commands that decode raw media.
    Domain-specific error subclasses can be supplied for backwards-compatible
    exception handling; all such errors share the MediaError base.
    """
    command = list(cmd)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=text,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise error_type(f"{label} timed out after {timeout:g} seconds.") from exc
    except OSError as exc:
        executable = command[0] if command else "media command"
        raise error_type(f"{label} could not start ({executable}): {exc}") from exc

    if result.returncode != 0:
        stderr = result.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        detail = str(stderr).strip()[-4000:] or "no diagnostic output"
        raise error_type(f"{label} failed with exit code {result.returncode}: {detail}")
    return result
