"""Local Ollama vision-model adapter."""

import base64
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class VisionModelError(RuntimeError):
    """Raised when the configured local vision model cannot analyze a frame."""


class OllamaVisionAnalyzer:
    """Analyze images with a locally running Ollama multimodal model."""

    def __init__(
        self,
        model: str = "qwen2.5vl:7b",
        base_url: str | None = None,
        timeout_seconds: int = 600,
    ) -> None:
        self.model = model
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        ).rstrip("/")
        self.timeout_seconds = timeout_seconds

    def analyze_frame(self, frame_path: Path) -> dict[str, Any]:
        """Return a normalized observation for one image."""
        image = base64.b64encode(frame_path.read_bytes()).decode("ascii")
        prompt = (
            "Analyze this travel-video frame. Return JSON only with exactly these keys: "
            '"description" (one factual sentence), "objects" (array of visible nouns), '
            '"activities" (array of directly visible actions), and "confidence" '
            "(number from 0 to 1). Do not guess the location, identities, or events "
            "that cannot be established from this frame. If uncertain, say so."
        )
        payload = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "format": "json",
                "messages": [{"role": "user", "content": prompt, "images": [image]}],
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
            raise VisionModelError("Ollama returned an invalid response.") from exc

        try:
            observation = json.loads(result["message"]["content"])
            description = str(observation["description"]).strip()
            objects = observation.get("objects", [])
            activities = observation.get("activities", [])
            confidence = float(observation.get("confidence", 0.5))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise VisionModelError(
                f"The vision model returned invalid structured output for {frame_path}."
            ) from exc
        if not description or not isinstance(objects, list) or not isinstance(activities, list):
            raise VisionModelError(f"The vision model returned malformed output for {frame_path}.")
        return {
            "description": description,
            "objects": [str(item) for item in objects],
            "activities": [str(item) for item in activities],
            "confidence": max(0.0, min(1.0, confidence)),
        }
