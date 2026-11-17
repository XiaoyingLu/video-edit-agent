"""Memory preflight and local model lifecycle helpers."""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class ResourceError(RuntimeError):
    """Raised when a model is unlikely to fit in the available memory budget."""


@dataclass(frozen=True)
class MemorySnapshot:
    """Best-effort memory available to the current container/process."""

    available_gb: float | None
    source: str


_VISION_MEMORY_GB = {"3": 6.0, "7": 12.0, "32": 48.0, "72": 96.0}
_WHISPER_MEMORY_GB = {
    "tiny": 1.0,
    "base": 1.5,
    "small": 3.0,
    "medium": 6.0,
    "large": 10.0,
    "large-v2": 10.0,
    "large-v3": 10.0,
}
_WHISPER_VRAM_GB = {
    "tiny": 1.5,
    "base": 2.0,
    "small": 3.0,
    "medium": 5.0,
    "large": 8.0,
    "large-v2": 8.0,
    "large-v3": 8.0,
}


def _read_mem_available_gb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _read_cgroup_available_gb() -> float | None:
    """Return remaining memory under a finite cgroup v1/v2 limit, if present."""
    candidates = (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    )
    for limit_path, usage_path in candidates:
        try:
            limit_text = limit_path.read_text(encoding="utf-8").strip()
            if limit_text == "max":
                continue
            limit = int(limit_text)
            usage = int(usage_path.read_text(encoding="utf-8").strip())
            # cgroup v1 uses a very large sentinel when no limit is configured.
            if limit <= 0 or limit >= 1 << 60:
                continue
            return max(0, limit - usage) / 1024**3
        except (OSError, ValueError):
            continue
    return None


def memory_snapshot() -> MemorySnapshot:
    """Use the lower of host-visible and finite container-available memory."""
    host_available = _read_mem_available_gb()
    cgroup_available = _read_cgroup_available_gb()
    known = [value for value in (host_available, cgroup_available) if value is not None]
    if not known:
        return MemorySnapshot(None, "unavailable")
    source = "host" if cgroup_available is None else "host/cgroup"
    return MemorySnapshot(min(known), source)


def required_vision_memory_gb(model: str) -> float:
    """Estimate a conservative RAM budget from a model tag."""
    override = os.getenv("OLLAMA_MIN_MEMORY_GB")
    if override:
        try:
            value = float(override)
        except ValueError as exc:
            raise ResourceError("OLLAMA_MIN_MEMORY_GB must be a positive number.") from exc
        if value <= 0:
            raise ResourceError("OLLAMA_MIN_MEMORY_GB must be a positive number.")
        return value
    match = re.search(r"(?<![0-9.])(3|7|32|72)b(?:$|[-_:])", model.lower())
    return _VISION_MEMORY_GB.get(match.group(1), 8.0) if match else 8.0


def required_whisper_memory_gb(model: str) -> float:
    """Estimate CPU RAM required for a faster-whisper model."""
    override = os.getenv("WHISPER_MIN_MEMORY_GB")
    if override:
        try:
            value = float(override)
        except ValueError as exc:
            raise ResourceError("WHISPER_MIN_MEMORY_GB must be a positive number.") from exc
        if value <= 0:
            raise ResourceError("WHISPER_MIN_MEMORY_GB must be a positive number.")
        return value
    return _WHISPER_MEMORY_GB.get(model.lower(), 4.0)


def required_whisper_vram_gb(model: str) -> float:
    """Estimate NVIDIA VRAM required for faster-whisper CUDA inference."""
    override = os.getenv("WHISPER_MIN_VRAM_GB")
    if override:
        try:
            value = float(override)
        except ValueError as exc:
            raise ResourceError("WHISPER_MIN_VRAM_GB must be a positive number.") from exc
        if value <= 0:
            raise ResourceError("WHISPER_MIN_VRAM_GB must be a positive number.")
        return value
    return _WHISPER_VRAM_GB.get(model.lower(), 4.0)


def available_gpu_memory_gb() -> float | None:
    """Return free NVIDIA VRAM when nvidia-smi is available."""
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
        values = [float(line.strip()) / 1024 for line in result.stdout.splitlines() if line.strip()]
        return values[0] if values else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def ensure_memory_available(required_gb: float, task: str) -> MemorySnapshot:
    """Fail early only when a measurable memory budget is below the estimate."""
    snapshot = memory_snapshot()
    if snapshot.available_gb is not None and snapshot.available_gb < required_gb:
        raise ResourceError(
            f"Not enough available memory for {task}: about {snapshot.available_gb:.1f} GB "
            f"available ({snapshot.source}), estimated minimum {required_gb:.1f} GB. "
            "Choose a smaller model or free memory. Override the estimate with the "
            "corresponding *_MIN_MEMORY_GB environment variable if appropriate."
        )
    return snapshot


def ensure_gpu_memory_available(required_gb: float, task: str) -> float | None:
    """Fail early when NVIDIA VRAM is measurable and below the estimate."""
    available = available_gpu_memory_gb()
    if available is not None and available < required_gb:
        raise ResourceError(
            f"Not enough free NVIDIA VRAM for {task}: {available:.1f} GB available, "
            f"estimated minimum {required_gb:.1f} GB. Use --device cpu or a smaller model."
        )
    return available
