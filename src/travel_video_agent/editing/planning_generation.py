"""LLM-backed narrative and storyboard generation for creative video plans."""

import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..resource_manager import ResourceError, ensure_memory_available, required_ollama_memory_gb
from .planning_schemas import PlannedScene, StoryTreatment


class GeneratedVideoPlanContent(BaseModel):
    """Creative content generated from a validated brief."""

    model_config = ConfigDict(extra="forbid")
    story_treatment: StoryTreatment
    scenes: list[PlannedScene] = Field(min_length=2, max_length=10)


class VideoPlanningError(RuntimeError):
    """Raised when the model cannot generate a valid narrative plan."""


def infer_creative_priorities(prompt: str, existing: list[str]) -> list[str]:
    """Supplement model-selected priorities with transparent keyword-based signals."""
    text = prompt.lower()
    keyword_groups = {
        "scenic_landscapes": (
            "mountain",
            "lake",
            "landscape",
            "scenery",
            "nature",
            "coast",
            "waterfall",
        ),
        "city_highlights": ("city", "street", "downtown", "urban"),
        "people_and_activities": (
            "people",
            "friends",
            "family",
            "hiking",
            "activity",
            "adventure",
        ),
        "food_and_culture": (
            "food",
            "restaurant",
            "cuisine",
            "culture",
            "market",
            "festival",
        ),
        "architecture": ("architecture", "building", "temple", "castle", "museum"),
    }
    priorities = list(existing)
    for priority, keywords in keyword_groups.items():
        if priority not in priorities and any(keyword in text for keyword in keywords):
            priorities.append(priority)
    return priorities


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
        self.model = model or os.getenv("VIDEO_PLANNER_MODEL", "qwen2.5:1.5b")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip(
            "/"
        )
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
        try:
            ensure_memory_available(
                required_ollama_memory_gb(self.model, "VIDEO_PLANNER_MIN_MEMORY_GB"),
                f"Ollama planning model '{self.model}'",
            )
        except ResourceError as exc:
            raise VideoPlanningError(str(exc)) from exc

    def unload_model(self) -> None:
        """Best-effort model eviction after planning."""
        payload = json.dumps({"model": self.model, "keep_alive": 0}).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=30):
                return
        except (HTTPError, URLError, TimeoutError, OSError):
            return

    def generate(
        self,
        prompt: str,
        brief: dict,
        music_sections: list[dict] | None = None,
        story_chapters: list[dict] | None = None,
    ) -> GeneratedVideoPlanContent:
        schema = GeneratedVideoPlanContent.model_json_schema()
        system_prompt = (
            "You are an experienced travel-film director and storyboard artist. "
            "Create a specific, coherent production storyboard from the creative brief. "
            "Do not merely restate the prompt. Include a compelling title, logline, "
            "beginning-middle-ending narrative arc, and 3-6 distinct scenes. "
            "Each scene must contain 2-5 concrete storyboard shots. Describe visible "
            "composition and subject, not generic phrases. Include shot scale, editorial "
            "role, purpose, optional concise narration/on-screen text, and useful audio cues. "
            "Use only supported enum values. The sum of all shot durations MUST equal the "
            "requested target duration exactly; each scene duration must equal the sum of "
            "its shot durations. Do not assign source footage or claim real footage has been "
            "inspected. For every storyboard shot, leave footage_path, "
            "footage_start_seconds, and footage_end_seconds unset, and set "
            "assignment_status to unassigned. Make choices grounded in the user's brief; "
            "flag uncertain details "
            "as assumptions rather than inventing personal trip events. When soundtrack "
            "energy sections are supplied, shape the scene progression to complement their "
            "low, medium, and high energy changes where editorially appropriate. Treat these "
            "as broad loudness cues, not semantic labels such as chorus or drop. Do not force "
            "every shot to end on a beat, and do not invent soundtrack events. When selected "
            "story chapters are supplied, create exactly one storyboard scene per chapter in "
            "the same order. Preserve each chapter name, purpose, mood, narration hint and "
            "music energy; distribute concrete shots within that chapter and make each scene "
            "duration match its chapter target."
        )
        payload = json.dumps(
            {
                "model": self.model,
                "keep_alive": self.keep_alive,
                "stream": False,
                "format": schema,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "request": prompt,
                                "brief": brief,
                                "soundtrack_energy_sections": music_sections or [],
                                "selected_story_chapters": story_chapters or [],
                            }
                        ),
                    },
                ],
                "options": {"temperature": 0.2},
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=300) as response:
                result = json.loads(response.read().decode("utf-8"))
            parsed = json.loads(result["message"]["content"])
            # Creative storyboards must never carry partial or speculative footage mappings.
            # Strip assignment fields before validation; footage mapping is a later stage.
            for scene in parsed.get("scenes", []):
                # Music-section annotations are derived from the analyzed timeline, not the LLM.
                scene.pop("music_energy", None)
                scene.pop("music_energy_sequence", None)
                scene.pop("music_section_start_seconds", None)
                scene.pop("music_section_end_seconds", None)
                scene.pop("music_alignment_note", None)
                for shot in scene.get("shots", []):
                    shot.pop("music_energy", None)
                    shot.pop("music_energy_sequence", None)
                    shot.pop("music_section_start_seconds", None)
                    shot.pop("music_section_end_seconds", None)
                    shot.pop("music_alignment_note", None)
                    shot.pop("pacing_recommendation", None)
                    shot.pop("pacing_rationale", None)
                    shot.pop("transition_recommendation", None)
                    shot.pop("transition_rationale", None)
                    shot.pop("footage_path", None)
                    shot.pop("footage_start_seconds", None)
                    shot.pop("footage_end_seconds", None)
                    shot.pop("assignment_status", None)
            return GeneratedVideoPlanContent.model_validate(parsed)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise VideoPlanningError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise VideoPlanningError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc
        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
            KeyError,
            TypeError,
            ValidationError,
        ) as exc:
            raise VideoPlanningError(
                f"Ollama returned an invalid story or storyboard: {exc}"
            ) from exc


def normalize_storyboard_ids(content: GeneratedVideoPlanContent) -> GeneratedVideoPlanContent:
    """Replace model-generated identifiers with deterministic, globally unique IDs."""
    normalized_scenes = []
    shot_number = 0
    for scene_number, scene in enumerate(content.scenes, start=1):
        scene_id = f"scene-{scene_number:02d}"
        shots = []
        for shot_number_in_scene, shot in enumerate(scene.shots, start=1):
            shot_number += 1
            shots.append(
                shot.model_copy(
                    update={
                        "shot_id": f"shot-{shot_number:03d}",
                        "scene_id": scene_id,
                        "sequence_index": shot_number_in_scene,
                    }
                )
            )
        normalized_scenes.append(
            scene.model_copy(
                update={
                    "scene_id": scene_id,
                    "sequence_index": scene_number,
                    "shots": shots,
                }
            )
        )
    return content.model_copy(update={"scenes": normalized_scenes})


def generate_video_plan_content(
    prompt: str,
    brief: dict,
    model: str | None = None,
    music_sections: list[dict] | None = None,
    story_chapters: list[dict] | None = None,
) -> GeneratedVideoPlanContent:
    """Generate structured story content and fit shot durations to the validated brief."""
    planner = OllamaVideoPlanner(model=model)
    try:
        content = planner.generate(
            prompt, brief, music_sections=music_sections, story_chapters=story_chapters
        )
    finally:
        planner.unload_model()

    content = normalize_storyboard_ids(content)
    all_shots = [shot for scene in content.scenes for shot in scene.shots]
    total = sum(shot.target_duration_seconds for shot in all_shots)
    if total <= 0:
        raise VideoPlanningError("Generated storyboard has no positive shot duration.")

    target = float(brief["target_duration_seconds"])
    if story_chapters is not None:
        if len(content.scenes) != len(story_chapters):
            raise VideoPlanningError(
                "Planner must return exactly one storyboard scene for each selected story chapter."
            )
        durations = []
        for scene, chapter in zip(content.scenes, story_chapters, strict=True):
            scene_total = sum(shot.target_duration_seconds for shot in scene.shots)
            if scene_total <= 0:
                raise VideoPlanningError("Generated chapter has no positive shot duration.")
            chapter_target = float(chapter["target_seconds"])
            chapter_durations = [
                round(shot.target_duration_seconds * chapter_target / scene_total, 2)
                for shot in scene.shots
            ]
            chapter_durations[-1] = round(chapter_target - sum(chapter_durations[:-1]), 2)
            durations.extend(chapter_durations)
        if abs(sum(float(item["target_seconds"]) for item in story_chapters) - target) > 0.05:
            raise VideoPlanningError("Selected story chapter durations do not match the creative brief.")
    else:
        factor = target / total
        durations = [round(shot.target_duration_seconds * factor, 2) for shot in all_shots]
        if durations:
            durations[-1] = round(target - sum(durations[:-1]), 2)
    if any(duration <= 0 or duration > 60 for duration in durations):
        raise VideoPlanningError(
            "Generated storyboard cannot fit the requested duration within shot limits."
        )

    normalized_scenes = []
    cursor = 0
    for scene in content.scenes:
        shots = []
        for shot in scene.shots:
            duration = durations[cursor]
            cursor += 1
            shots.append(shot.model_copy(update={"target_duration_seconds": duration}))
        normalized_scenes.append(
            scene.model_copy(
                update={
                    "shots": shots,
                    "target_duration_seconds": round(
                        sum(shot.target_duration_seconds for shot in shots), 2
                    ),
                }
            )
        )
    return content.model_copy(update={"scenes": normalized_scenes})
