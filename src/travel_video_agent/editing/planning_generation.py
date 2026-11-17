"""LLM-backed narrative and storyboard generation for creative video plans."""

import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..resource_manager import ResourceError, ensure_memory_available, required_vision_memory_gb
from .planning_schemas import PlannedScene, StoryTreatment


class GeneratedVideoPlanContent(BaseModel):
    """Creative content generated from a validated brief."""

    model_config = ConfigDict(extra="forbid")
    story_treatment: StoryTreatment
    scenes: list[PlannedScene] = Field(min_length=2, max_length=10)


class VideoPlanningError(RuntimeError):
    """Raised when the model cannot generate a valid narrative plan."""


def explicit_duration_seconds(prompt: str) -> float | None:
    """Extract an explicitly requested duration so model defaults cannot override it."""
    text = prompt.lower()
    patterns = (
        (r"\b(\d+(?:\.\d+)?)\s*(?:-| )?seconds?\b", 1.0),
        (r"\b(\d+(?:\.\d+)?)\s*(?:-| )?minutes?\b", 60.0),
        (r"\b(\d+(?:\.\d+)?)\s*(?:-| )?hours?\b", 3600.0),
    )
    matches = []
    for pattern, multiplier in patterns:
        for match in re.finditer(pattern, text):
            matches.append((match.start(), float(match.group(1)) * multiplier))
    return matches[0][1] if matches else None


class OllamaVideoPlanner:
    """Generate a narrative treatment and storyboard using Ollama structured output."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
        try:
            ensure_memory_available(required_vision_memory_gb(self.model), f"Ollama planning model '{self.model}'")
        except ResourceError as exc:
            raise VideoPlanningError(str(exc)) from exc

    def unload_model(self) -> None:
        """Best-effort model eviction after planning."""
        payload = json.dumps({"model": self.model, "keep_alive": 0}).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(request, timeout=30):
                return
        except (HTTPError, URLError, TimeoutError, OSError):
            return

    def generate(self, prompt: str, brief: dict) -> GeneratedVideoPlanContent:
        schema = GeneratedVideoPlanContent.model_json_schema()
        system_prompt = (
            "You are an experienced travel-film director and storyboard artist. Create a specific, "
            "coherent production storyboard from the creative brief. Do not merely restate the prompt. "
            "Include a compelling title, logline, beginning-middle-ending narrative arc, and 3-6 "
            "distinct scenes. Each scene must contain 2-5 concrete storyboard shots. Describe visible "
            "composition and subject, not generic phrases. Include shot scale, editorial role, purpose, "
            "optional concise narration/on-screen text, and useful audio cues. Use only supported enum "
            "values. The sum of all shot durations MUST equal the requested target duration exactly; "
            "each scene duration must equal the sum of its shot durations. Do not assign source footage. "
            "Do not claim real footage has been inspected. Make choices grounded in the user's brief; "
            "flag uncertain details as assumptions rather than inventing personal trip events."
        )
        payload = json.dumps({
            "model": self.model,
            "keep_alive": self.keep_alive,
            "stream": False,
            "format": schema,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps({"request": prompt, "brief": brief})},
            ],
            "options": {"temperature": 0.2},
        }).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/chat", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(request, timeout=300) as response:
                result = json.loads(response.read().decode("utf-8"))
            parsed = json.loads(result["message"]["content"])
            content = GeneratedVideoPlanContent.model_validate(parsed)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise VideoPlanningError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise VideoPlanningError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValidationError) as exc:
            raise VideoPlanningError(f"Ollama returned an invalid story or storyboard: {exc}") from exc
        return content


def generate_video_plan_content(prompt: str, brief: dict, model: str | None = None) -> GeneratedVideoPlanContent:
    """Generate structured story content; normalize durations to the validated brief."""
    planner = OllamaVideoPlanner(model=model)
    try:
        content = planner.generate(prompt, brief)
    finally:
        planner.unload_model()
    total = sum(shot.target_duration_seconds for scene in content.scenes for shot in scene.shots)
    if total <= 0:
        raise VideoPlanningError("Generated storyboard has no positive shot duration.")
    target = float(brief["target_duration_seconds"])
    factor = target / total
    normalized_scenes = []
    all_shots = [(scene, shot) for scene in content.scenes for shot in scene.shots]
    durations = [round(shot.target_duration_seconds * factor, 2) for _, shot in all_shots]
    if durations:
        durations[-1] = round(target - sum(durations[:-1]), 2)
    cursor = 0
    for scene in content.scenes:
        shots = []
        for shot in scene.shots:
            duration = durations[cursor]
            cursor += 1
            shots.append(shot.model_copy(update={"target_duration_seconds": duration}))
        normalized_scenes.append(scene.model_copy(update={
            "shots": shots,
            "target_duration_seconds": round(sum(s.target_duration_seconds for s in shots), 2),
        }))
    return content.model_copy(update={"scenes": normalized_scenes})
