"""Centralized environment-backed application settings."""

from __future__ import annotations

import os

DEFAULT_OLLAMA_MODEL = "qwen2.5vl:3b"
DEFAULT_TITLE_FONT = "Lato"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_KEEP_ALIVE = "5m"
DEFAULT_VIDEO_PLANNER_MODEL = "qwen2.5:1.5b"
DEFAULT_VIDEO_STORY_NUM_CTX = "8192"
DEFAULT_VISION_PROVIDER = "ollama"
DEFAULT_QWEN3_VL_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
DEFAULT_MLX_VISION_MODEL = "mlx-community/SmolVLM2-2.2B-Instruct-mlx"

SETTING_DEFAULTS: dict[str, str | None] = {
    "OLLAMA_MODEL": DEFAULT_OLLAMA_MODEL,
    "TITLE_FONT": DEFAULT_TITLE_FONT,
    "OLLAMA_BASE_URL": DEFAULT_OLLAMA_BASE_URL,
    "OLLAMA_KEEP_ALIVE": DEFAULT_OLLAMA_KEEP_ALIVE,
    "VIDEO_PLANNER_MODEL": DEFAULT_VIDEO_PLANNER_MODEL,
    "VIDEO_STORY_NUM_CTX": DEFAULT_VIDEO_STORY_NUM_CTX,
    "VISION_PROVIDER": DEFAULT_VISION_PROVIDER,
    "QWEN3_VL_MODEL": DEFAULT_QWEN3_VL_MODEL,
    "MLX_VISION_MODEL": DEFAULT_MLX_VISION_MODEL,
    "OLLAMA_MIN_MEMORY_GB": None,
    "WHISPER_MIN_MEMORY_GB": None,
    "WHISPER_MIN_VRAM_GB": None,
    "FOOTAGE_EMBEDDING_MODEL": None,
    "SOUNDSTRIPE_API_KEY": None,
    "STABILITY_API_KEY": None,
}


def get_setting(name: str, default: str | None = None) -> str | None:
    """Read an environment setting, using its centralized default when defined."""
    return os.getenv(name, SETTING_DEFAULTS.get(name) if default is None else default)
