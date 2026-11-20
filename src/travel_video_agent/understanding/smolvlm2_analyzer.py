"""Experimental SmolVLM2 vision adapter for native Apple Silicon via mlx-vlm."""

import gc
import json
from pathlib import Path
from typing import Any

from .schemas import SceneDynamics
from .visual_analyzer import VisionModelError, _parse_observation

_FRAME_PROMPT = """Analyze this image as a travel-film editor. Return exactly one compact JSON object.
Use exactly these keys and no others:
{"description":"one short factual sentence","objects":["up to 5 important visible things"],"activities":["up to 2 visible actions"],"confidence":0.0,"shot_scale":"wide","camera_angle":"eye_level","composition":"unknown","lighting":"daylight","mood":"unknown","visual_role":"unknown","setting":"broad setting","primary_subject":"main visible subject"}
Allowed shot_scale: extreme_wide, wide, medium, close_up, extreme_close_up, aerial, unknown.
Allowed camera_angle: eye_level, low_angle, high_angle, birds_eye, worms_eye, unknown.
Allowed composition: centered, rule_of_thirds, leading_lines, symmetrical, layered_depth, negative_space, unknown.
Allowed lighting: golden_hour, blue_hour, daylight, overcast, low_light, artificial, backlit, mixed, unknown.
Allowed mood: serene, energetic, dramatic, romantic, mysterious, joyful, contemplative, tense, unknown.
Allowed visual_role: establishing, action, detail, reaction, transition, portrait, scenic_breath, unknown.
Use short values. Describe visible evidence only; never guess a precise location, identity, event, or camera movement. Use unknown when uncertain. Output valid JSON only, on one line. No Markdown or commentary."""

_TEMPORAL_PROMPT = """Analyze these ordered frames from one continuous travel-video scene.
The frames are supplied in chronological order. Describe only changes supported by
the sequence. Distinguish subject movement from camera movement. Use camera_motion
unknown unless viewpoint changes clearly support pan, tilt, zoom, tracking, or handheld
movement. Changing subjects alone is not camera motion. Do not invent events between
sampled frames. Return only a JSON object with temporal_summary (<=300 characters),
action_progression (up to 6 strings), subject_motion (approaching, receding, crossing,
stationary, interacting, mixed, unknown), camera_motion (static, pan, tilt, zoom,
tracking, handheld, mixed, unknown), and confidence (0..1)."""


def _json_object(text: str, *, label: str) -> dict[str, Any]:
    """Parse model JSON, tolerating a surrounding Markdown fence or short preamble."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1]
        if candidate.endswith("```"):
            candidate = candidate[:-3].strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        raise VisionModelError(f"SmolVLM2 returned no JSON object for {label}.")
    try:
        result = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise VisionModelError(f"SmolVLM2 returned invalid JSON for {label}: {exc}") from exc
    if not isinstance(result, dict):
        raise VisionModelError(f"SmolVLM2 response for {label} must be a JSON object.")
    return result


class SmolVLM2MLXAnalyzer:
    """Run SmolVLM2 locally using MLX-VLM's native Apple Silicon backend.

    This adapter must run in a native macOS Python process; Docker Desktop's Linux
    containers cannot access Apple's Metal GPU.
    """

    def __init__(
        self,
        model: str = "mlx-community/SmolVLM2-2.2B-Instruct-mlx",
        *,
        max_tokens: int = 1024,
    ) -> None:
        self.model_id = model
        self.max_tokens = max_tokens
        self._model: Any = None
        self._processor: Any = None
        self._config: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
        except ImportError as exc:
            raise VisionModelError(
                'The MLX provider requires mlx-vlm. Install it with `pip install -e ".[mlx]"` '
                "in native macOS Python."
            ) from exc
        try:
            self._model, self._processor = load(self.model_id)
            self._config = load_config(self.model_id)
        except Exception as exc:
            raise VisionModelError(
                f"Could not load MLX vision model '{self.model_id}': {exc}"
            ) from exc

    def _generate(self, prompt: str, images: list[Path], *, max_tokens: int | None = None) -> str:
        self._load()
        try:
            from mlx_vlm import generate
            from mlx_vlm.prompt_utils import apply_chat_template

            image_input = [str(path) for path in images]
            formatted_prompt = apply_chat_template(
                self._processor, self._config, prompt, num_images=len(image_input)
            )
            result = generate(
                self._model,
                self._processor,
                formatted_prompt,
                image_input,
                max_tokens=max_tokens or self.max_tokens,
                temp=0.0,
            )
        except Exception as exc:
            raise VisionModelError(
                "SmolVLM2 MLX inference failed. Confirm that the selected model supports "
                f"the installed mlx-vlm version and image input: {exc}"
            ) from exc
        if isinstance(result, str):
            return result
        text = getattr(result, "text", None)
        if isinstance(text, str):
            return text
        raise VisionModelError("MLX-VLM returned an unsupported generation result.")

    def analyze_frame(self, frame_path: Path) -> dict[str, Any]:
        """Return the same normalized observation shape as the Ollama analyzer."""
        if not frame_path.is_file():
            raise VisionModelError(f"Frame does not exist: {frame_path}")
        content = self._generate(_FRAME_PROMPT, [frame_path])
        try:
            return _parse_observation(content, frame_path)
        except VisionModelError as first_error:
            retry_prompt = (
                f"{_FRAME_PROMPT}\nIMPORTANT: Your previous response was incomplete or "
                "invalid. Return one complete, syntactically valid JSON object. Keep arrays "
                "within the stated limits, use short values, and close every bracket. "
                "Do not add any text before or after the JSON."
            )
            retry_content = self._generate(
                retry_prompt, [frame_path], max_tokens=max(self.max_tokens * 2, 2048)
            )
            try:
                return _parse_observation(retry_content, frame_path)
            except VisionModelError as retry_error:
                raise VisionModelError(
                    f"SmolVLM2 returned invalid frame output after retry for {frame_path}. "
                    f"First attempt: {first_error} Retry attempt: {retry_error}"
                ) from retry_error

    def analyze_scene(self, frames: list[tuple[float, Path]]) -> dict[str, Any]:
        """Analyze ordered scene frames together to infer bounded temporal dynamics."""
        if len(frames) < 2:
            return SceneDynamics().model_dump()
        selected = frames
        if len(selected) > 6:
            indices = [round(index * (len(selected) - 1) / 5) for index in range(6)]
            selected = [selected[index] for index in indices]
        paths = [path for _, path in selected]
        timestamps = ", ".join(f"{timestamp:.2f}s" for timestamp, _ in selected)
        prompt = f"{_TEMPORAL_PROMPT}\nFrame timestamps: {timestamps}."
        content = self._generate(prompt, paths)
        try:
            dynamics = _json_object(content, label="temporal analysis")
            return SceneDynamics.model_validate(dynamics).model_dump()
        except Exception as first_error:
            retry_prompt = (
                f"{prompt}\nIMPORTANT: Return a complete, valid JSON object only. "
                "Use short values, obey all array limits, and close every bracket."
            )
            retry_content = self._generate(
                retry_prompt, paths, max_tokens=max(self.max_tokens * 2, 1024)
            )
            try:
                dynamics = _json_object(retry_content, label="temporal analysis")
                return SceneDynamics.model_validate(dynamics).model_dump()
            except Exception as retry_error:
                raise VisionModelError(
                    "SmolVLM2 returned invalid temporal output after retry. "
                    f"First attempt: {first_error} Retry attempt: {retry_error}"
                ) from retry_error

    def unload_model(self) -> bool:
        """Release model references and clear the MLX Metal cache."""
        self._model = None
        self._processor = None
        self._config = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, AttributeError):
            pass
        return True
