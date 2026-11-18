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
        "shot_scale": {
            "type": "string",
            "enum": [
                "extreme_wide",
                "wide",
                "medium",
                "close_up",
                "extreme_close_up",
                "aerial",
                "unknown",
            ],
        },
        "camera_angle": {
            "type": "string",
            "enum": ["eye_level", "low_angle", "high_angle", "birds_eye", "worms_eye", "unknown"],
        },
        "composition": {
            "type": "string",
            "enum": [
                "centered",
                "rule_of_thirds",
                "leading_lines",
                "symmetrical",
                "layered_depth",
                "negative_space",
                "unknown",
            ],
        },
        "lighting": {
            "type": "string",
            "enum": [
                "golden_hour",
                "blue_hour",
                "daylight",
                "overcast",
                "low_light",
                "artificial",
                "backlit",
                "mixed",
                "unknown",
            ],
        },
        "mood": {
            "type": "string",
            "enum": [
                "serene",
                "energetic",
                "dramatic",
                "romantic",
                "mysterious",
                "joyful",
                "contemplative",
                "tense",
                "unknown",
            ],
        },
        "visual_role": {
            "type": "string",
            "enum": [
                "establishing",
                "action",
                "detail",
                "reaction",
                "transition",
                "portrait",
                "scenic_breath",
                "unknown",
            ],
        },
        "setting": {"type": "string", "maxLength": 100},
        "primary_subject": {"type": "string", "maxLength": 100},
    },
    "required": [
        "description",
        "objects",
        "activities",
        "confidence",
        "shot_scale",
        "camera_angle",
        "composition",
        "lighting",
        "mood",
        "visual_role",
        "setting",
        "primary_subject",
    ],
    "additionalProperties": False,
}


_STORYBOARD_MATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "match": {"type": "string", "enum": ["match", "partial", "mismatch", "uncertain"]},
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string", "maxLength": 240},
        "suggestion": {"type": "string", "maxLength": 240},
    },
    "required": ["match", "score", "confidence", "evidence", "suggestion"],
    "additionalProperties": False,
}


def _enum_value(value: Any, allowed: set[str]) -> str:
    """Normalize optional editorial labels; unknown values never leak into artifacts."""
    return value if isinstance(value, str) and value in allowed else "unknown"


def _short_text(value: Any) -> str:
    """Keep optional scene labels bounded and safe for downstream prompts/UI."""
    return value.strip()[:100] if isinstance(value, str) else ""


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
        "shot_scale": _enum_value(
            observation.get("shot_scale"),
            {
                "extreme_wide",
                "wide",
                "medium",
                "close_up",
                "extreme_close_up",
                "aerial",
                "unknown",
            },
        ),
        "camera_angle": _enum_value(
            observation.get("camera_angle"),
            {"eye_level", "low_angle", "high_angle", "birds_eye", "worms_eye", "unknown"},
        ),
        "composition": _enum_value(
            observation.get("composition"),
            {
                "centered",
                "rule_of_thirds",
                "leading_lines",
                "symmetrical",
                "layered_depth",
                "negative_space",
                "unknown",
            },
        ),
        "lighting": _enum_value(
            observation.get("lighting"),
            {
                "golden_hour",
                "blue_hour",
                "daylight",
                "overcast",
                "low_light",
                "artificial",
                "backlit",
                "mixed",
                "unknown",
            },
        ),
        "mood": _enum_value(
            observation.get("mood"),
            {
                "serene",
                "energetic",
                "dramatic",
                "romantic",
                "mysterious",
                "joyful",
                "contemplative",
                "tense",
                "unknown",
            },
        ),
        "visual_role": _enum_value(
            observation.get("visual_role"),
            {
                "establishing",
                "action",
                "detail",
                "reaction",
                "transition",
                "portrait",
                "scenic_breath",
                "unknown",
            },
        ),
        "setting": _short_text(observation.get("setting")),
        "primary_subject": _short_text(observation.get("primary_subject")),
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
            "Analyze this travel-video frame as a professional travel-film editor. Return one JSON "
            "object matching the supplied schema. Describe only visible evidence; never invent a "
            "specific location, identity, event, or camera movement. Be concise: description must "
            "be one short factual sentence (200 characters or fewer); list at most 8 important "
            "objects and 4 visible activities. Classify shot scale, camera angle, composition, "
            "lighting, mood, and likely editorial role (establishing/action/detail/reaction/"
            "transition/portrait/scenic_breath). Use unknown when uncertain. Infer setting and "
            "primary subject only at a broad visual level. A single still frame cannot reliably "
            "prove camera movement, so do not infer it. Avoid generic praise and subjective "
            "quality claims.",
            "Return concise factual JSON matching the schema. Include one short description, at "
            "most 5 objects and 2 activities. Classify shot scale, angle, composition, lighting, "
            "mood, editorial role, broad setting, and primary subject. Use unknown whenever "
            "uncertain; do not guess location or camera movement.",
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

    def compare_frame_to_storyboard(
        self,
        frame_path: Path,
        *,
        purpose: str,
        visual_description: str,
        primary_subject: str = "",
        expected_role: str = "unknown",
        expected_scale: str = "unknown",
    ) -> dict[str, Any]:
        """Compare one rendered frame with its intended storyboard shot."""
        if not self._memory_checked:
            try:
                ensure_memory_available(
                    required_vision_memory_gb(self.model), f"Ollama vision model '{self.model}'"
                )
            except ResourceError as exc:
                raise VisionModelError(str(exc)) from exc
            self._memory_checked = True
        image = base64.b64encode(frame_path.read_bytes()).decode("ascii")
        prompt = (
            "Act as a careful travel-film continuity editor. Compare the supplied rendered "
            "video frame with the intended storyboard shot below. Judge only whether the visible "
            "content supports the intended subject, action, setting, shot purpose, and broad "
            "composition. Do not penalize harmless differences in lighting, framing, or exact "
            "camera angle. Do not assume the image shows events outside this single frame. "
            "Use 'uncertain' and low confidence when evidence is ambiguous or the frame is a "
            "transition. A scenic establishing shot can satisfy a scenic purpose without people. "
            "Return concise, evidence-based JSON; never invent details.\n"
            f"Purpose: {purpose[:240]}\n"
            f"Intended visual: {visual_description[:600]}\n"
            f"Primary subject: {primary_subject[:100] or 'not specified'}\n"
            f"Expected editorial role: {expected_role}\n"
            f"Expected shot scale: {expected_scale}"
        )
        payload = json.dumps(
            {
                "model": self.model,
                "keep_alive": self.keep_alive,
                "stream": False,
                "format": _STORYBOARD_MATCH_SCHEMA,
                "messages": [{"role": "user", "content": prompt, "images": [image]}],
                "options": {"temperature": 0, "num_predict": 180},
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
                self._model_used = True
            assessment = json.loads(result["message"]["content"])
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise VisionModelError(
                f"Ollama returned HTTP {exc.code} during storyboard comparison: {detail}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise VisionModelError(
                f"Could not reach Ollama at {self.base_url} for storyboard comparison."
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as exc:
            raise VisionModelError("Ollama returned invalid storyboard comparison JSON.") from exc

        if not isinstance(assessment, dict):
            raise VisionModelError("Storyboard comparison must be a JSON object.")
        match = assessment.get("match")
        score = assessment.get("score")
        confidence = assessment.get("confidence")
        evidence = assessment.get("evidence")
        suggestion = assessment.get("suggestion")
        if (
            not isinstance(match, str)
            or match not in {"match", "partial", "mismatch", "uncertain"}
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or not isinstance(evidence, str)
            or len(evidence) > 240
            or not isinstance(suggestion, str)
            or len(suggestion) > 240
        ):
            raise VisionModelError("Ollama returned invalid storyboard comparison fields.")
        return {
            "match": match,
            "score": float(score),
            "confidence": float(confidence),
            "evidence": evidence.strip(),
            "suggestion": suggestion.strip(),
        }

    def analyze_scene(self, frames: list[tuple[float, Path]]) -> dict[str, Any]:
        """Infer temporal action and camera movement from ordered scene frames."""
        if len(frames) < 2:
            return {
                "temporal_summary": "",
                "action_progression": [],
                "subject_motion": "unknown",
                "camera_motion": "unknown",
                "confidence": 0.0,
            }
        if not self._memory_checked:
            try:
                ensure_memory_available(
                    required_vision_memory_gb(self.model), f"Ollama vision model '{self.model}'"
                )
            except ResourceError as exc:
                raise VisionModelError(str(exc)) from exc
            self._memory_checked = True

        selected = frames
        if len(selected) > 6:
            indices = [round(index * (len(selected) - 1) / 5) for index in range(6)]
            selected = [selected[index] for index in indices]
        images = [
            base64.b64encode(frame_path.read_bytes()).decode("ascii") for _, frame_path in selected
        ]
        timestamps = ", ".join(f"{timestamp:.2f}s" for timestamp, _ in selected)
        response_schema = {
            "type": "object",
            "properties": {
                "temporal_summary": {"type": "string", "maxLength": 300},
                "action_progression": {
                    "type": "array",
                    "items": {"type": "string", "maxLength": 100},
                    "maxItems": 6,
                },
                "subject_motion": {
                    "type": "string",
                    "enum": [
                        "approaching",
                        "receding",
                        "crossing",
                        "stationary",
                        "interacting",
                        "mixed",
                        "unknown",
                    ],
                },
                "camera_motion": {
                    "type": "string",
                    "enum": [
                        "static",
                        "pan",
                        "tilt",
                        "zoom",
                        "tracking",
                        "handheld",
                        "mixed",
                        "unknown",
                    ],
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": [
                "temporal_summary",
                "action_progression",
                "subject_motion",
                "camera_motion",
                "confidence",
            ],
            "additionalProperties": False,
        }
        prompt = (
            "Analyze these ordered frames from one continuous travel-video scene. "
            f"Frame timestamps: {timestamps}. Describe only changes supported by the sequence. "
            "Distinguish subject movement from camera movement. Use camera_motion 'unknown' "
            "unless viewpoint changes clearly support pan, tilt, zoom, tracking, or handheld "
            "movement. Changing subjects alone is not camera motion. Do not invent events "
            "between sampled frames. Return concise JSON matching the schema."
        )
        payload = json.dumps(
            {
                "model": self.model,
                "keep_alive": self.keep_alive,
                "stream": False,
                "format": response_schema,
                "messages": [{"role": "user", "content": prompt, "images": images}],
                "options": {"temperature": 0, "num_predict": 220},
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
                self._model_used = True
            dynamics = json.loads(result["message"]["content"])
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise VisionModelError(
                f"Ollama returned HTTP {exc.code} during temporal scene analysis: {detail}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise VisionModelError(
                f"Could not reach Ollama at {self.base_url} for temporal scene analysis."
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as exc:
            raise VisionModelError("Ollama returned invalid temporal scene analysis JSON.") from exc

        if not isinstance(dynamics, dict):
            raise VisionModelError("Temporal scene analysis must be a JSON object.")
        summary = dynamics.get("temporal_summary")
        progression = dynamics.get("action_progression")
        confidence = dynamics.get("confidence")
        if (
            not isinstance(summary, str)
            or len(summary) > 300
            or not isinstance(progression, list)
            or len(progression) > 6
            or any(not isinstance(item, str) or len(item) > 100 for item in progression)
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise VisionModelError("Ollama returned invalid temporal scene analysis fields.")
        return {
            "temporal_summary": summary.strip(),
            "action_progression": [item.strip() for item in progression],
            "subject_motion": _enum_value(
                dynamics.get("subject_motion"),
                {
                    "approaching",
                    "receding",
                    "crossing",
                    "stationary",
                    "interacting",
                    "mixed",
                    "unknown",
                },
            ),
            "camera_motion": _enum_value(
                dynamics.get("camera_motion"),
                {
                    "static",
                    "pan",
                    "tilt",
                    "zoom",
                    "tracking",
                    "handheld",
                    "mixed",
                    "unknown",
                },
            ),
            "confidence": float(confidence),
        }

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
