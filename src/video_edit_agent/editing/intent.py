"""Natural-language editing intent parsing through a local Ollama model."""

import warnings

from pydantic import ValidationError

from ..config import get_setting
from ..resource_manager import (
    ResourceError,
    ensure_memory_available,
    required_ollama_memory_gb,
)
from .intent_schemas import EditingPlan
from .ollama_client import (
    CHAT_TIMEOUT_SECONDS,
    CONTROL_TIMEOUT_SECONDS,
    ModelClient,
    ModelClientError,
    OllamaClient,
)


class IntentParsingError(RuntimeError):
    """Raised when the local model cannot produce a valid editing plan."""


class OllamaIntentParser:
    """Convert a user prompt into a constrained, validated editing plan."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout_seconds: int = CHAT_TIMEOUT_SECONDS,
        memory_override_env: str = "OLLAMA_MIN_MEMORY_GB",
        client: ModelClient | None = None,
    ) -> None:
        self.model = model or get_setting("OLLAMA_MODEL")
        self.client = client if client is not None else OllamaClient(base_url=base_url)
        self.timeout_seconds = timeout_seconds
        self._model_used = False
        if isinstance(self.client, OllamaClient):
            try:
                ensure_memory_available(
                    required_ollama_memory_gb(self.model, memory_override_env),
                    f"Ollama intent model '{self.model}',",
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
        try:
            parsed = self.client.chat_json(
                self.model,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": normalized_prompt},
                ],
                schema,
                options={"temperature": 0},
                timeout_seconds=self.timeout_seconds,
            )
            self._model_used = True
        except ModelClientError as exc:
            raise IntentParsingError(str(exc)) from exc

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
        if not self._model_used:
            return True
        try:
            self.client.generate(
                self.model,
                timeout_seconds=CONTROL_TIMEOUT_SECONDS,
                keep_alive=0,
            )
            return True
        except ModelClientError:
            return False


def parse_editing_intent(
    prompt: str,
    model: str | None = None,
    *,
    memory_override_env: str = "OLLAMA_MIN_MEMORY_GB",
    client: ModelClient | None = None,
) -> EditingPlan:
    """Parse one natural-language request, then release the model from memory."""
    parser = OllamaIntentParser(
        model=model,
        memory_override_env=memory_override_env,
        client=client,
    )
    try:
        return parser.parse(prompt)
    finally:
        if not parser.unload_model():
            warnings.warn(
                f"Could not unload Ollama model '{parser.model}'; it may remain "
                "resident in memory.",
                RuntimeWarning,
                stacklevel=2,
            )
