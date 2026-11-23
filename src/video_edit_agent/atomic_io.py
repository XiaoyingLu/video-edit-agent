"""Crash-safe JSON persistence helpers."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any
from uuid import uuid4


def write_json_atomic(path: Path, model: Any) -> None:
    """Serialize a model or JSON value and atomically replace the destination.

    The temporary file is created beside the destination so replacement stays
    on one filesystem. Existing permissions are retained; new files use the
    process umask just like Path.write_text.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    dump = getattr(model, "model_dump", None)
    payload = dump(mode="json") if callable(dump) else model
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    try:
        existing_mode = stat.S_IMODE(destination.stat().st_mode)
    except FileNotFoundError:
        existing_mode = None

    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        if existing_mode is not None:
            os.chmod(temporary, existing_mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
