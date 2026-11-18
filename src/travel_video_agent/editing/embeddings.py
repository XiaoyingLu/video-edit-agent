"""Small, local Ollama embedding client used for semantic footage retrieval."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EmbeddingModelError(RuntimeError):
    """Raised when the local embedding model cannot produce valid vectors."""


def ollama_embed_texts(
    texts: list[str],
    model: str,
    *,
    base_url: str | None = None,
    timeout_seconds: int = 180,
) -> list[list[float]]:
    """Embed a batch of texts through Ollama's local /api/embed endpoint."""
    if not texts:
        return []
    endpoint = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    request = Request(
        f"{endpoint}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise EmbeddingModelError(
            f"Ollama returned HTTP {exc.code} while embedding footage text: {detail}. "
            f"Check that embedding model '{model}' is installed."
        ) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise EmbeddingModelError(
            f"Could not reach Ollama at {endpoint} for footage embeddings. "
            "Start the Ollama service and verify OLLAMA_BASE_URL."
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EmbeddingModelError("Ollama returned an invalid embedding response.") from exc

    vectors = result.get("embeddings") if isinstance(result, dict) else None
    if (
        not isinstance(vectors, list)
        or len(vectors) != len(texts)
        or any(
            not isinstance(vector, list)
            or not vector
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in vector
            )
            for vector in vectors
        )
    ):
        raise EmbeddingModelError(
            f"Ollama embedding model '{model}' returned an invalid vector batch."
        )
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise EmbeddingModelError("Ollama returned embeddings with inconsistent dimensions.")
    return [[float(value) for value in vector] for vector in vectors]
