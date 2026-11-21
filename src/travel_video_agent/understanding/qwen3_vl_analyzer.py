"""Production Qwen3-VL adapter for native Apple Silicon via MLX-VLM."""

from pathlib import Path
from typing import Any

from .schemas import SceneDynamics
from .smolvlm2_analyzer import (
    _TEMPORAL_PROMPT,
    SmolVLM2MLXAnalyzer,
    _json_object,
    _string_array,
)
from .visual_analyzer import VisionModelError

DEFAULT_QWEN3_VL_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"

_SUBJECT_MOTION = {
    "approaching",
    "receding",
    "crossing",
    "stationary",
    "interacting",
    "mixed",
    "unknown",
}
_CAMERA_MOTION = {"static", "pan", "tilt", "zoom", "tracking", "handheld", "mixed", "unknown"}


def _classify_motion(value: Any, *, field: str) -> str:
    """Map Qwen's descriptive motion labels to the bounded production enum."""
    allowed = _SUBJECT_MOTION if field == "subject_motion" else _CAMERA_MOTION
    if not isinstance(value, str):
        return "unknown"
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in allowed:
        return normalized
    if field == "subject_motion":
        if any(word in normalized for word in ("stationary", "standing_still", "not_moving")):
            return "stationary"
        if any(word in normalized for word in ("approach", "toward_camera", "coming_closer")):
            return "approaching"
        if any(word in normalized for word in ("reced", "moving_away", "away_from_camera")):
            return "receding"
        if any(word in normalized for word in ("cross", "running", "walking", "moving_across")):
            return "crossing"
        if any(word in normalized for word in ("interact", "acrobat", "juggling", "with_hoop")):
            return "interacting"
    else:
        if "handheld" in normalized or "shaky" in normalized:
            return "handheld"
        if "track" in normalized or "follow" in normalized:
            return "tracking"
        if "pan" in normalized:
            return "pan"
        if "tilt" in normalized:
            return "tilt"
        if "zoom" in normalized:
            return "zoom"
        if "static" in normalized or "fixed" in normalized:
            return "static"
    return "unknown"


def _normalize_motion(value: Any, *, field: str) -> str:
    """Normalize Qwen scalar/list motion descriptions to one schema enum."""
    values = value if isinstance(value, list) else [value]
    classified = {_classify_motion(item, field=field) for item in values if isinstance(item, str)}
    classified.discard("unknown")
    if not classified:
        return "unknown"
    if len(classified) > 1:
        return "mixed"
    return next(iter(classified))


def _normalize_temporal_output(raw: str) -> dict[str, Any]:
    """Normalize Qwen's occasional list-valued enum fields before strict validation."""
    payload = _json_object(raw, label="temporal analysis")
    payload["action_progression"] = _string_array(payload.get("action_progression"))[:6]
    for field in ("subject_motion", "camera_motion"):
        payload[field] = _normalize_motion(payload.get(field), field=field)
    return payload


class Qwen3VLMLXAnalyzer(SmolVLM2MLXAnalyzer):
    """Analyze frames and temporal scene dynamics with Qwen3-VL on Apple Silicon."""

    provider_name = "Qwen3-VL"

    def __init__(
        self,
        model: str = DEFAULT_QWEN3_VL_MODEL,
        *,
        max_tokens: int = 1024,
    ) -> None:
        super().__init__(model=model, max_tokens=max_tokens)

    def analyze_scene(self, frames: list[tuple[float, Path]]) -> dict[str, Any]:
        """Analyze all ordered representative frames for bounded temporal dynamics."""
        if len(frames) < 2:
            return SceneDynamics().model_dump()

        paths = [path for _, path in frames]
        timestamps = ", ".join(f"{timestamp:.2f}s" for timestamp, _ in frames)
        prompt = f"{_TEMPORAL_PROMPT}\nFrame timestamps: {timestamps}."
        content = self._generate(prompt, paths)
        try:
            return SceneDynamics.model_validate(_normalize_temporal_output(content)).model_dump()
        except (VisionModelError, ValueError, TypeError, KeyError) as first_error:
            retry_prompt = (
                f"{prompt}\nIMPORTANT: Return one complete valid JSON object only. "
                "subject_motion and camera_motion must each be a single allowed enum value, "
                "not an array. Keep action_progression to at most six strings."
            )
            retry_content = self._generate(
                retry_prompt, paths, max_tokens=max(self.max_tokens * 2, 2048)
            )
            try:
                return SceneDynamics.model_validate(
                    _normalize_temporal_output(retry_content)
                ).model_dump()
            except (VisionModelError, ValueError, TypeError, KeyError) as retry_error:
                raise VisionModelError(
                    "Qwen3-VL returned invalid temporal output after retry. "
                    f"First attempt: {first_error} Retry attempt: {retry_error}"
                ) from retry_error
