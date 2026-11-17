"""Natural-language editing intent parsing through a local Ollama model."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from ..resource_manager import ResourceError, ensure_memory_available, required_vision_memory_gb
from .intent_schemas import EditingPlan


class IntentParsingError(RuntimeError):
    """Raised when the local model cannot produce a valid editing plan."""


class OllamaIntentParser:
    """Convert a user prompt into a constrained, validated editing plan."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = 180,
    ) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip(
            "/"
        )
        self.timeout_seconds = timeout_seconds
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
        try:
            ensure_memory_available(
                required_vision_memory_gb(self.model), f"Ollama intent model '{self.model}'"
            )
        except ResourceError as exc:
            raise IntentParsingError(str(exc)) from exc

    def parse(self, prompt: str) -> EditingPlan:
        """Ask Ollama for structured intent, then validate it locally."""
        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise ValueError("editing prompt must not be empty")
        if len(normalized_prompt) > 4000:
            raise ValueError("editing prompt must be 4000 characters or fewer")

        schema = EditingPlan.model_json_schema()
        schema["properties"].pop("source_prompt", None)
        schema["properties"].pop("schema_version", None)
        schema["properties"].pop("approval_required", None)
        schema["properties"].pop("rendering_performed", None)
        schema["required"] = [
            field
            for field in schema.get("required", [])
            if field
            not in {"source_prompt", "schema_version", "approval_required", "rendering_performed"}
        ]
        system_prompt = (
            "Translate the user's video editing request into the supplied JSON schema. "
            "Return only schema-conforming JSON. Do not invent footage contents or claim "
            "that editing has been performed. Use safe defaults for unspecified options. "
            "Only use the enumerated values. If the user asks for an unsupported operation, "
            "do not add it; record a concise explanation in assumptions. "
            "A request to remove silence means propose speech-aware review candidates, "
            "never delete quiet scenic footage automatically. For cut transitions, set "
            "duration_seconds to 0."
        )
        payload = json.dumps(
            {
                "model": self.model,
                "keep_alive": self.keep_alive,
                "stream": False,
                "format": schema,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": normalized_prompt},
                ],
                "options": {"temperature": 0},
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
            content = result["message"]["content"]
            parsed = json.loads(content)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise IntentParsingError(
                f"Ollama returned HTTP {exc.code}: {detail}. "
                f"Check that model '{self.model}' is installed."
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise IntentParsingError(
                f"Could not reach Ollama at {self.base_url}. Start the Docker service "
                "and verify OLLAMA_BASE_URL."
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as exc:
            raise IntentParsingError("Ollama returned an invalid structured response.") from exc

        if not isinstance(parsed, dict):
            raise IntentParsingError("Ollama returned JSON, but the response was not an object.")
        try:
            return EditingPlan.model_validate(
                {
                    **parsed,
                    "source_prompt": normalized_prompt,
                    "schema_version": "1.0",
                    "approval_required": True,
                    "rendering_performed": False,
                }
            )
        except ValidationError as exc:
            raise IntentParsingError(
                f"The model response did not satisfy the editing-plan schema: {exc}"
            ) from exc

    def unload_model(self) -> bool:
        """Ask Ollama to evict the intent model before another model phase begins."""
        payload = json.dumps({"model": self.model, "keep_alive": 0}).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=min(self.timeout_seconds, 30)):
                return True
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
            return False


def parse_editing_intent(prompt: str, model: str | None = None) -> EditingPlan:
    """Parse one natural-language request, then release the model from memory."""
    parser = OllamaIntentParser(model=model)
    try:
        return parser.parse(prompt)
    finally:
        parser.unload_model()
