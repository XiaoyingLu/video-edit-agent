"""Shared model-client interface and Ollama HTTP implementation.

A hosted provider can implement ModelClient without changing editing workflows.
"""

from __future__ import annotations

import json
import time
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..config import get_setting

CHAT_TIMEOUT_SECONDS = 300
LONG_CONTEXT_CHAT_TIMEOUT_SECONDS = 600
EMBED_TIMEOUT_SECONDS = 180
CONTROL_TIMEOUT_SECONDS = 30
MAX_RETRIES = 1
RETRY_DELAY_SECONDS = 0.5
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class ModelClientError(RuntimeError):
    """Raised when Ollama cannot return a valid model response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        endpoint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.endpoint = endpoint


# Kept as an alias for callers that name the concrete Ollama implementation.
OllamaClientError = ModelClientError


class ModelClient(Protocol):
    """Provider-neutral interface used by model-backed application workflows."""

    def chat_json(
        self,
        model: str,
        messages: list[dict],
        schema: dict,
        *,
        options: dict | None = None,
        timeout_seconds: int = CHAT_TIMEOUT_SECONDS,
        keep_alive: str | int | None = None,
    ) -> dict: ...

    def generate(
        self,
        model: str,
        prompt: str = "",
        *,
        options: dict | None = None,
        timeout_seconds: int = CONTROL_TIMEOUT_SECONDS,
        keep_alive: str | int | None = None,
    ) -> dict: ...

    def embed(
        self,
        model: str,
        texts: list[str],
        *,
        timeout_seconds: int = EMBED_TIMEOUT_SECONDS,
    ) -> list[list[float]]: ...


class OllamaClient:
    """Ollama implementation with centralized configuration, parsing, and retries."""

    def __init__(
        self,
        base_url: str | None = None,
        keep_alive: str | int | None = None,
        *,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self.base_url = (base_url or get_setting("OLLAMA_BASE_URL")).rstrip("/")
        self.keep_alive = keep_alive if keep_alive is not None else get_setting("OLLAMA_KEEP_ALIVE")
        self.max_retries = max_retries

    def _request(self, endpoint: str, payload: dict, timeout_seconds: int) -> dict:
        request = Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=timeout_seconds) as response:
                    result = json.loads(response.read().decode("utf-8"))
                if not isinstance(result, dict):
                    raise ModelClientError(
                        f"Ollama returned a non-object response from {endpoint}.",
                        endpoint=endpoint,
                    )
                return result
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code in _RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue
                raise ModelClientError(
                    f"Ollama returned HTTP {exc.code} from {endpoint}: {detail}",
                    status_code=exc.code,
                    endpoint=endpoint,
                ) from exc
            except (URLError, TimeoutError, OSError) as exc:
                if attempt < self.max_retries:
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue
                raise ModelClientError(
                    f"Could not reach Ollama at {self.base_url}{endpoint}: {exc}",
                    endpoint=endpoint,
                ) from exc
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ModelClientError(
                    f"Ollama returned invalid JSON from {endpoint}: {exc}",
                    endpoint=endpoint,
                ) from exc
        raise ModelClientError(f"Ollama request failed: {endpoint}", endpoint=endpoint)

    def chat_json(
        self,
        model: str,
        messages: list[dict],
        schema: dict,
        *,
        options: dict | None = None,
        timeout_seconds: int = CHAT_TIMEOUT_SECONDS,
        keep_alive: str | int | None = None,
    ) -> dict:
        """Call /api/chat with JSON-schema output and parse message.content."""
        payload = self._request(
            "/api/chat",
            {
                "model": model,
                "keep_alive": self.keep_alive if keep_alive is None else keep_alive,
                "stream": False,
                "format": schema,
                "messages": messages,
                "options": options or {},
            },
            timeout_seconds,
        )
        content = None
        try:
            content = payload["message"]["content"]
            result = json.loads(content)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            preview = " ".join(content.split())[:240] if isinstance(content, str) else repr(content)
            raise ModelClientError(
                f"Ollama returned an invalid structured chat response: {exc}. "
                f"Response preview: {preview!r}",
                endpoint="/api/chat",
            ) from exc
        if not isinstance(result, dict):
            raise ModelClientError(
                "Ollama returned structured JSON, but it was not an object.",
                endpoint="/api/chat",
            )
        return result

    def generate(
        self,
        model: str,
        prompt: str = "",
        *,
        options: dict | None = None,
        timeout_seconds: int = CONTROL_TIMEOUT_SECONDS,
        keep_alive: str | int | None = None,
    ) -> dict:
        """Call /api/generate; also supports keep_alive=0 model eviction."""
        payload = {
            "model": model,
            "keep_alive": self.keep_alive if keep_alive is None else keep_alive,
        }
        if prompt:
            payload["prompt"] = prompt
            payload["stream"] = False
        if options is not None:
            payload["options"] = options
        return self._request("/api/generate", payload, timeout_seconds)

    def embed(
        self,
        model: str,
        texts: list[str],
        *,
        timeout_seconds: int = EMBED_TIMEOUT_SECONDS,
    ) -> list[list[float]]:
        """Call /api/embed and validate a complete, consistent vector batch."""
        if not texts:
            return []
        payload = self._request(
            "/api/embed",
            {"model": model, "input": texts, "keep_alive": self.keep_alive},
            timeout_seconds,
        )
        vectors = payload.get("embeddings")
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
            raise ModelClientError(
                f"Ollama embedding model '{model}' returned an invalid vector batch.",
                endpoint="/api/embed",
            )
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise ModelClientError(
                "Ollama returned embeddings with inconsistent dimensions.",
                endpoint="/api/embed",
            )
        return [[float(value) for value in vector] for vector in vectors]
