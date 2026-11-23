"""Embedding adapter backed by the shared model-client interface."""

from .ollama_client import EMBED_TIMEOUT_SECONDS, ModelClient, OllamaClient, ModelClientError


class EmbeddingModelError(RuntimeError):
    """Raised when the embedding provider cannot produce valid vectors."""


def ollama_embed_texts(
    texts: list[str],
    model: str,
    *,
    base_url: str | None = None,
    timeout_seconds: int = EMBED_TIMEOUT_SECONDS,
    client: ModelClient | None = None,
) -> list[list[float]]:
    """Embed a batch of texts through the shared provider interface."""
    if not texts:
        return []
    model_client = client if client is not None else OllamaClient(base_url=base_url)
    try:
        return model_client.embed(model, texts, timeout_seconds=timeout_seconds)
    except ModelClientError as exc:
        raise EmbeddingModelError(str(exc)) from exc
