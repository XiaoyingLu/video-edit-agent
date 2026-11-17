"""Local Ollama vision-model adapter."""

import base64
import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..resource_manager import ResourceError, ensure_memory_available, required_vision_memory_gb


class VisionModelError(RuntimeError):
    """Raised when the configured local vision model cannot analyze a frame."""


_VISION_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "maxLength": 200},
        "objects": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "activities": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["description", "objects", "activities", "confidence"],
    "additionalProperties": False,
}


def _parse_observation(content: Any, frame_path: Path) -> dict[str, Any]:
    """Parse and validate model content, including common Markdown-wrapped JSON."""
    if not isinstance(content, str) or not content.strip():
        raise VisionModelError(
            f"The vision model returned empty or non-text output for {frame_path}."
        )

    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        observation = json.loads(text)
    except json.JSONDecodeError as exc:
        preview = " ".join(content.split())[:240]
        raise VisionModelError(
            f"The vision model returned invalid JSON for {frame_path} "
            f"(line {exc.lineno}, column {exc.colno}). Response preview: {preview!r}"
        ) from exc

    if not isinstance(observation, dict):
        raise VisionModelError(f"The vision model response for {frame_path} must be a JSON object.")

    description = observation.get("description")
    objects = observation.get("objects")
    activities = observation.get("activities")
    confidence = observation.get("confidence")

    if not isinstance(description, str) or not description.strip():
        raise VisionModelError(
            f"The vision model response for {frame_path} is missing a non-empty description."
        )
    if len(description) > 200:
        raise VisionModelError(
            f"The vision model description for {frame_path} exceeds 200 characters."
        )
    if not isinstance(objects, list) or not all(isinstance(item, str) for item in objects):
        raise VisionModelError(
            f"The vision model response for {frame_path} must contain an objects string array."
        )
    if len(objects) > 8:
        raise VisionModelError(
            f"The vision model response for {frame_path} contains more than 8 objects."
        )
    if not isinstance(activities, list) or not all(isinstance(item, str) for item in activities):
        raise VisionModelError(
            f"The vision model response for {frame_path} must contain an activities string array."
        )
    if len(activities) > 4:
        raise VisionModelError(
            f"The vision model response for {frame_path} contains more than 4 activities."
        )
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise VisionModelError(
            f"The vision model response for {frame_path} must contain confidence between 0 and 1."
        )

    return {
        "description": description.strip(),
        "objects": objects,
        "activities": activities,
        "confidence": float(confidence),
    }


class OllamaVisionAnalyzer:
    """Analyze images with a locally running Ollama multimodal model."""

    def __init__(
        self,
        model: str = "qwen2.5vl:7b",
        base_url: str | None = None,
        timeout_seconds: int = 600,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip(
            "/"
        )
        self.timeout_seconds = timeout_seconds
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
        self._memory_checked = False
        self._model_used = False

    def analyze_frame(self, frame_path: Path) -> dict[str, Any]:
        """Return a normalized observation, retrying once if the model emits invalid JSON."""
        if not self._memory_checked:
            try:
                ensure_memory_available(
                    required_vision_memory_gb(self.model), f"Ollama vision model '{self.model}'"
                )
            except ResourceError as exc:
                raise VisionModelError(str(exc)) from exc
            self._memory_checked = True
        image = base64.b64encode(frame_path.read_bytes()).decode("ascii")
        prompts = (
            "Analyze this travel-video frame. Return one JSON object matching the supplied "
            "schema. Be concise: description must be one short factual sentence (200 characters "
            "or fewer); list at most 8 important visible objects and at most 4 directly visible "
            "activities. Do not enumerate minor details or repeat similar objects. Do not guess "
            "locations, identities, or events. Use empty arrays when appropriate.",
            "Return a very concise JSON object matching the schema. Include one short factual "
            "description, no more than 5 key objects, no more than 2 visible activities, and "
            "confidence from 0 to 1. Do not add explanations or extra details.",
        )
        previous_error: VisionModelError | None = None

        for attempt, prompt in enumerate(prompts):
            payload = json.dumps(
                {
                    "model": self.model,
                    "keep_alive": self.keep_alive,
                    "stream": False,
                    "format": _VISION_RESPONSE_SCHEMA,
                    "messages": [{"role": "user", "content": prompt, "images": [image]}],
                    "options": {"temperature": 0, "num_predict": 256 if attempt == 0 else 192},
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
                    raw_response = response.read()
                    self._model_used = True
                    result = json.loads(raw_response.decode("utf-8"))
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise VisionModelError(
                    f"Ollama returned HTTP {exc.code}: {detail}. "
                    f"Check that model '{self.model}' is installed."
                ) from exc
            except (URLError, TimeoutError) as exc:
                raise VisionModelError(
                    f"Could not reach Ollama at {self.base_url}. Start the Docker service "
                    "and verify OLLAMA_BASE_URL."
                ) from exc
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as exc:
                raise VisionModelError("Ollama returned an invalid response envelope.") from exc

            try:
                content = result["message"]["content"]
            except (KeyError, TypeError) as exc:
                raise VisionModelError(
                    f"Ollama response for {frame_path} did not contain message.content."
                ) from exc

            try:
                return _parse_observation(content, frame_path)
            except VisionModelError as exc:
                if attempt == 0:
                    previous_error = exc
                    continue
                raise VisionModelError(
                    f"The vision model returned invalid output for {frame_path} after one "
                    f"short-output retry. First attempt: {previous_error}"
                ) from exc

        raise VisionModelError(f"The vision model could not analyze {frame_path}.")


    def unload_model(self) -> bool:
        """Ask Ollama to evict this model from VRAM/RAM after the vision phase."""
        if not self._model_used:
            return True
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
            # A completed analysis must not be discarded just because best-effort eviction failed.
            return False
