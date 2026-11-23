"""Vision analyzer provider selection."""

from ..config import get_setting
from .qwen3_vl_analyzer import Qwen3VLMLXAnalyzer
from .smolvlm2_analyzer import SmolVLM2MLXAnalyzer
from .visual_analyzer import OllamaVisionAnalyzer


def create_vision_analyzer(provider: str, model: str | None = None):
    """Create the requested analyzer without importing optional MLX dependencies eagerly."""
    selected = provider.strip().lower()
    if selected == "ollama":
        return OllamaVisionAnalyzer(model=model or get_setting("OLLAMA_MODEL"))
    if selected == "qwen3-vl":
        return Qwen3VLMLXAnalyzer(
            model=model or get_setting("QWEN3_VL_MODEL")
        )
    if selected == "mlx":
        return SmolVLM2MLXAnalyzer(
            model=model or get_setting("MLX_VISION_MODEL")
        )
    raise ValueError(
        f"Unsupported vision provider '{provider}'. Choose 'ollama', 'mlx', or 'qwen3-vl'."
    )
