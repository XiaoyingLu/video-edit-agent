"""Vision analyzer provider selection."""

import os

from .qwen3_vl_analyzer import Qwen3VLMLXAnalyzer
from .smolvlm2_analyzer import SmolVLM2MLXAnalyzer
from .visual_analyzer import OllamaVisionAnalyzer


def create_vision_analyzer(provider: str, model: str | None = None):
    """Create the requested analyzer without importing optional MLX dependencies eagerly."""
    selected = provider.strip().lower()
    if selected == "ollama":
        return OllamaVisionAnalyzer(model=model or os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b"))
    if selected == "qwen3-vl":
        return Qwen3VLMLXAnalyzer(
            model=model or os.getenv("QWEN3_VL_MODEL", "mlx-community/Qwen3-VL-4B-Instruct-4bit")
        )
    if selected == "mlx":
        return SmolVLM2MLXAnalyzer(
            model=model or os.getenv("MLX_VISION_MODEL", "mlx-community/SmolVLM2-2.2B-Instruct-mlx")
        )
    raise ValueError(
        f"Unsupported vision provider '{provider}'. Choose 'ollama', 'mlx', or 'qwen3-vl'."
    )
