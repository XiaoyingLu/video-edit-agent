import pytest

from travel_video_agent import resource_manager as resources


def test_vision_model_memory_estimates_and_override(monkeypatch):
    monkeypatch.delenv("OLLAMA_MIN_MEMORY_GB", raising=False)
    assert resources.required_vision_memory_gb("qwen2.5vl:3b") == 6.0
    assert resources.required_vision_memory_gb("qwen2.5vl:7b") == 12.0

    monkeypatch.setenv("OLLAMA_MIN_MEMORY_GB", "9.5")
    assert resources.required_vision_memory_gb("qwen2.5vl:7b") == 9.5


def test_whisper_memory_estimates_and_override(monkeypatch):
    monkeypatch.delenv("WHISPER_MIN_MEMORY_GB", raising=False)
    monkeypatch.delenv("WHISPER_MIN_VRAM_GB", raising=False)
    assert resources.required_whisper_memory_gb("small") == 3.0
    assert resources.required_whisper_vram_gb("small") == 3.0

    monkeypatch.setenv("WHISPER_MIN_MEMORY_GB", "2.25")
    monkeypatch.setenv("WHISPER_MIN_VRAM_GB", "2.75")
    assert resources.required_whisper_memory_gb("small") == 2.25
    assert resources.required_whisper_vram_gb("small") == 2.75


def test_memory_preflight_fails_before_model_load(monkeypatch):
    monkeypatch.setattr(
        resources, "memory_snapshot", lambda: resources.MemorySnapshot(2.0, "test cgroup")
    )
    with pytest.raises(resources.ResourceError, match="2.0 GB available"):
        resources.ensure_memory_available(6.0, "Qwen 2.5 VL 3B")


def test_memory_preflight_allows_unknown_memory(monkeypatch):
    monkeypatch.setattr(
        resources, "memory_snapshot", lambda: resources.MemorySnapshot(None, "unavailable")
    )
    snapshot = resources.ensure_memory_available(12.0, "Qwen 2.5 VL 7B")
    assert snapshot.available_gb is None


def test_gpu_preflight_rejects_insufficient_vram(monkeypatch):
    monkeypatch.setattr(resources, "available_gpu_memory_gb", lambda: 1.0)
    with pytest.raises(resources.ResourceError, match="Not enough free NVIDIA VRAM"):
        resources.ensure_gpu_memory_available(3.0, "Whisper small")


def test_gpu_preflight_is_optional_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(resources, "available_gpu_memory_gb", lambda: None)
    assert resources.ensure_gpu_memory_available(8.0, "Whisper large") is None


@pytest.mark.parametrize(
    ("variable", "resolver"),
    [
        ("OLLAMA_MIN_MEMORY_GB", resources.required_vision_memory_gb),
        ("WHISPER_MIN_MEMORY_GB", resources.required_whisper_memory_gb),
        ("WHISPER_MIN_VRAM_GB", resources.required_whisper_vram_gb),
    ],
)
def test_resource_threshold_rejects_non_finite_values(monkeypatch, variable, resolver):
    monkeypatch.setenv(variable, "nan")
    with pytest.raises(resources.ResourceError, match="positive number"):
        resolver("qwen2.5vl:7b" if variable.startswith("OLLAMA") else "small")
