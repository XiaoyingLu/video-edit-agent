"""Footage-first story proposals built from the deterministic story ledger.

The proposer reads what was actually shot (via the story ledger) and suggests a few
alternative narrative angles. Every scene ID a model returns is validated against the
ledger, so hallucinated IDs are dropped before anything is saved. Proposals are advisory:
nothing here assigns footage, approves a plan, edits media, or renders video.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .artifact_tracking import register_artifact, validate_artifact
from .editing.planning_generation import explicit_duration_seconds
from .highlights.schemas import ProjectHighlights
from .models import ProjectAnalysis
from .resource_manager import ResourceError, ensure_memory_available, required_ollama_memory_gb
from .story_ledger import StoryLedger, build_story_ledger
from .transcription.schemas import ProjectTranscription
from .understanding.schemas import VideoUnderstanding

DEFAULT_TARGET_SECONDS = 90.0
DEFAULT_ANGLE_COUNT = 2
DEFAULT_NUM_CTX = 8192
STORY_PROPOSAL_FILENAME = "story-proposal.json"


class StoryProposalError(RuntimeError):
    """Raised when a story proposal cannot be generated or validated."""


class StoryChapter(BaseModel):
    """One narrative beat, grounded in real scene IDs from the ledger."""

    model_config = ConfigDict(extra="forbid")

    chapter_id: str = Field(min_length=1, max_length=40)
    name: str = Field(min_length=1, max_length=120)
    purpose: str = Field(min_length=1, max_length=300)
    mood: str = Field(default="", max_length=80)
    target_seconds: float = Field(gt=0, le=3600)
    candidate_scene_ids: list[str] = Field(min_length=1, max_length=40)
    must_include_scene_ids: list[str] = Field(default_factory=list, max_length=10)
    narration_hint: str = Field(default="", max_length=400)
    music_energy: Literal["low", "medium", "high", "unknown"] = "unknown"


class UnusedScene(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    reason: str = Field(default="", max_length=200)


class StoryAngle(BaseModel):
    """One alternative way to tell the story with the available footage."""

    model_config = ConfigDict(extra="forbid")

    angle_id: str = Field(min_length=1, max_length=40)
    title: str = Field(min_length=1, max_length=120)
    logline: str = Field(min_length=1, max_length=400)
    story_angle: str = Field(min_length=1, max_length=80)
    chapters: list[StoryChapter] = Field(min_length=2, max_length=8)
    unused_scenes: list[UnusedScene] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list, max_length=8)
    warnings: list[str] = Field(default_factory=list)


class GeneratedStoryAngles(BaseModel):
    """Raw model output, validated structurally before ledger grounding."""

    model_config = ConfigDict(extra="forbid")

    angles: list[StoryAngle] = Field(min_length=1, max_length=4)


class StoryProposal(BaseModel):
    """Saved artifact; the human picks an angle with select_story_angle."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    project_path: str
    source_prompt: str | None = None
    target_duration_seconds: float = Field(gt=0, le=3600)
    planner_model: str
    ledger_scene_count: int = Field(ge=0)
    angles: list[StoryAngle] = Field(min_length=1)
    selected_angle_id: str | None = None
    approval_required: Literal[True] = True
    rendering_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_selection(self) -> StoryProposal:
        ids = [angle.angle_id for angle in self.angles]
        if len(ids) != len(set(ids)):
            raise ValueError("angle IDs must be unique")
        if self.selected_angle_id is not None and self.selected_angle_id not in ids:
            raise ValueError("selected angle does not exist")
        return self


class StoryModel(Protocol):
    """Anything that can turn a prompt and JSON schema into a JSON object."""

    model: str

    def generate_json(self, system_prompt: str, user_content: str, schema: dict) -> dict: ...


class OllamaStoryModel:
    """Structured-output story generation through a local Ollama text model."""

    def __init__(self, model: str | None = None, base_url: str | None = None) -> None:
        self.model = model or os.getenv("VIDEO_PLANNER_MODEL", "qwen2.5:1.5b")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip(
            "/"
        )
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "5m")
        self.num_ctx = int(os.getenv("VIDEO_STORY_NUM_CTX", str(DEFAULT_NUM_CTX)))
        try:
            ensure_memory_available(
                required_ollama_memory_gb(self.model, "VIDEO_PLANNER_MIN_MEMORY_GB"),
                f"Ollama story model '{self.model}'",
            )
        except ResourceError as exc:
            raise StoryProposalError(str(exc)) from exc

    def unload_model(self) -> None:
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

    def generate_json(self, system_prompt: str, user_content: str, schema: dict) -> dict:
        # Ollama silently truncates input beyond num_ctx, which would drop part of the
        # ledger without any error. Fail loudly instead (about 3.5 characters per token).
        estimated_tokens = (len(system_prompt) + len(user_content)) / 3.5 + 1500
        if estimated_tokens > self.num_ctx:
            raise StoryProposalError(
                f"Ledger needs roughly {int(estimated_tokens)} tokens but the context window is "
                f"{self.num_ctx}. Raise VIDEO_STORY_NUM_CTX (and use a model that supports it) "
                "or analyze fewer/shorter videos."
            )
        payload = json.dumps(
            {
                "model": self.model,
                "keep_alive": self.keep_alive,
                "stream": False,
                "format": schema,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "options": {"temperature": 0.3, "num_ctx": self.num_ctx},
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=600) as response:
                result = json.loads(response.read().decode("utf-8"))
            return json.loads(result["message"]["content"])
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise StoryProposalError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise StoryProposalError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError) as exc:
            raise StoryProposalError(f"Ollama returned unreadable story JSON: {exc}") from exc


SYSTEM_PROMPT = (
    "You are an editor who builds a story from footage that has already been shot. "
    "You are given a STORY LEDGER: scenes grouped into clusters, each line starting with a "
    "scene ID. Propose {angle_count} clearly different story angles (for example a "
    "chronological journey, a best-moments montage, or a mood piece). For each angle give a "
    "title, a logline, and 3-6 chapters in narrative order. Each chapter must list "
    "candidate_scene_ids copied EXACTLY from the ledger; never invent or alter scene IDs. "
    "Use spoken lines, moods, settings and highlight scores as evidence, and do not claim "
    "events that the ledger does not show. Chapter target_seconds must add up to the target "
    "duration. Put footage you would leave out in unused_scenes with a short reason, and list "
    "in gaps any story beats the footage cannot support (for example no strong closing shot). "
    "Leave warnings empty. Use stable IDs such as angle-1 and ch-1."
)


def _user_content(ledger: StoryLedger, target_seconds: float, hint: str | None) -> str:
    parts = [f"Target duration: {target_seconds:g} seconds."]
    if hint:
        parts.append(f"Editor's hint (tone or focus, not a scene list): {hint}")
    parts.append(ledger.render())
    return "\n\n".join(parts)


def _scene_durations(ledger: StoryLedger) -> dict[str, float]:
    return {entry.scene_id: entry.end_seconds - entry.start_seconds for entry in ledger.entries}


def ground_angle(angle: StoryAngle, ledger: StoryLedger, target_seconds: float) -> StoryAngle:
    """Drop hallucinated or duplicate scene IDs, fit durations, and record warnings."""
    known = ledger.scene_ids
    durations = _scene_durations(ledger)
    warnings = list(angle.warnings)
    claimed: set[str] = set()
    chapters: list[StoryChapter] = []

    for chapter in angle.chapters:
        candidates: list[str] = []
        for scene_id in chapter.candidate_scene_ids:
            if scene_id not in known:
                warnings.append(f"{chapter.chapter_id}: dropped unknown scene ID '{scene_id}'.")
            elif scene_id in claimed or scene_id in candidates:
                warnings.append(f"{chapter.chapter_id}: scene {scene_id} was already used.")
            else:
                candidates.append(scene_id)
        if not candidates:
            raise StoryProposalError(
                f"Angle '{angle.angle_id}' chapter '{chapter.chapter_id}' has no valid scenes."
            )
        claimed.update(candidates)
        must = [scene_id for scene_id in chapter.must_include_scene_ids if scene_id in candidates]
        if len(must) != len(chapter.must_include_scene_ids):
            warnings.append(f"{chapter.chapter_id}: removed must-include scenes not in candidates.")
        chapters.append(
            chapter.model_copy(
                update={"candidate_scene_ids": candidates, "must_include_scene_ids": must}
            )
        )

    total = sum(chapter.target_seconds for chapter in chapters)
    scale = target_seconds / total
    scaled = [round(chapter.target_seconds * scale, 2) for chapter in chapters]
    scaled[-1] = round(target_seconds - sum(scaled[:-1]), 2)
    if scaled[-1] <= 0:
        raise StoryProposalError(f"Angle '{angle.angle_id}' cannot fit the target duration.")
    fitted: list[StoryChapter] = []
    for chapter, seconds in zip(chapters, scaled, strict=True):
        available = sum(durations[scene_id] for scene_id in chapter.candidate_scene_ids)
        if available < seconds:
            warnings.append(
                f"{chapter.chapter_id}: candidate footage totals {available:.1f}s, "
                f"less than the {seconds:.1f}s target."
            )
        fitted.append(chapter.model_copy(update={"target_seconds": seconds}))

    unused = []
    for item in angle.unused_scenes:
        if item.scene_id not in known:
            warnings.append(f"Ignored unknown unused scene ID '{item.scene_id}'.")
        elif item.scene_id in claimed:
            warnings.append(f"Scene {item.scene_id} is both used and listed as unused.")
        else:
            unused.append(item)
    return angle.model_copy(
        update={"chapters": fitted, "unused_scenes": unused, "warnings": warnings}
    )


def generate_story_angles(
    ledger: StoryLedger,
    story_model: StoryModel,
    *,
    target_seconds: float,
    angle_count: int = DEFAULT_ANGLE_COUNT,
    hint: str | None = None,
) -> list[StoryAngle]:
    """Ask the model for angles, then ground each one against the ledger."""
    if not ledger.entries:
        raise StoryProposalError("No analyzed scenes; run 'understand' before proposing a story.")
    raw = story_model.generate_json(
        SYSTEM_PROMPT.format(angle_count=angle_count),
        _user_content(ledger, target_seconds, hint),
        GeneratedStoryAngles.model_json_schema(),
    )
    try:
        generated = GeneratedStoryAngles.model_validate(raw)
    except ValidationError as exc:
        raise StoryProposalError(f"Model returned an invalid story proposal: {exc}") from exc

    angles: list[StoryAngle] = []
    skipped: list[str] = []
    for number, angle in enumerate(generated.angles[:angle_count], start=1):
        angle = angle.model_copy(update={"angle_id": f"angle-{number}"})
        try:
            angles.append(ground_angle(angle, ledger, target_seconds))
        except StoryProposalError as exc:
            skipped.append(str(exc))
    if not angles:
        raise StoryProposalError("No usable story angle: " + " ".join(skipped))
    return angles


def _load_optional(root: Path, path: Path, schema: type[BaseModel]):
    if not path.is_file():
        return None
    validate_artifact(root, path)
    return schema.model_validate_json(path.read_text(encoding="utf-8"))


def propose_story(
    folder: Path,
    *,
    prompt: str | None = None,
    target_seconds: float | None = None,
    angle_count: int = DEFAULT_ANGLE_COUNT,
    model: str | None = None,
    output_path: Path | None = None,
    story_model: StoryModel | None = None,
) -> tuple[StoryProposal, Path]:
    """Build the ledger from analyzed footage and save grounded story angles."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project folder does not exist: {root}")
    if not 1 <= angle_count <= 4:
        raise ValueError("angle count must be between 1 and 4")
    artifact_dir = root / ".video-agent"
    analysis_file = artifact_dir / "analysis.json"
    understanding_file = artifact_dir / "understanding.json"
    for required, command in ((analysis_file, "analyze"), (understanding_file, "understand")):
        if not required.is_file():
            raise FileNotFoundError(f"Missing {required}; run 'video-agent {command}' first.")

    analysis = _load_optional(root, analysis_file, ProjectAnalysis)
    understanding = _load_optional(root, understanding_file, VideoUnderstanding)
    transcription_file = artifact_dir / "transcription.json"
    highlights_file = artifact_dir / "highlights.json"
    transcription = _load_optional(root, transcription_file, ProjectTranscription)
    highlights = _load_optional(root, highlights_file, ProjectHighlights)

    destination = (output_path or artifact_dir / STORY_PROPOSAL_FILENAME).expanduser().resolve()
    if destination in {analysis_file, understanding_file, transcription_file, highlights_file}:
        raise ValueError("story proposal output cannot overwrite an input artifact")

    target = target_seconds
    if target is None and prompt:
        target = explicit_duration_seconds(prompt)
    target = target or DEFAULT_TARGET_SECONDS
    if not 0 < target <= 3600:
        raise ValueError("target duration must be between 0 and 3600 seconds")

    ledger = build_story_ledger(
        root, understanding, analysis, transcription=transcription, highlights=highlights
    )
    owned_model = story_model is None
    story_model = story_model or OllamaStoryModel(model=model)
    try:
        angles = generate_story_angles(
            ledger, story_model, target_seconds=target, angle_count=angle_count, hint=prompt
        )
    finally:
        if owned_model and isinstance(story_model, OllamaStoryModel):
            story_model.unload_model()

    proposal = StoryProposal(
        project_path=str(root),
        source_prompt=prompt,
        target_duration_seconds=target,
        planner_model=story_model.model,
        ledger_scene_count=len(ledger.entries),
        angles=angles,
    )
    _write(
        root,
        destination,
        proposal,
        dependencies=[
            path
            for path in (analysis_file, understanding_file, transcription_file, highlights_file)
            if path.is_file()
        ],
    )
    return proposal, destination


def select_story_angle(
    folder: Path, angle_id: str, *, proposal_path: Path | None = None
) -> tuple[StoryProposal, Path]:
    """Record the human's choice of angle without regenerating anything."""
    root = folder.expanduser().resolve()
    path = (proposal_path or root / ".video-agent" / STORY_PROPOSAL_FILENAME).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run 'video-agent propose-story' first.")
    validate_artifact(root, path)
    proposal = StoryProposal.model_validate_json(path.read_text(encoding="utf-8"))
    if angle_id not in {angle.angle_id for angle in proposal.angles}:
        choices = ", ".join(angle.angle_id for angle in proposal.angles)
        raise ValueError(f"Unknown angle '{angle_id}'. Choose one of: {choices}")
    proposal = proposal.model_copy(update={"selected_angle_id": angle_id})
    # dependencies=None preserves the existing upstream edges.
    _write(root, path, proposal, dependencies=None)
    return proposal, path


def _write(
    root: Path, path: Path, proposal: StoryProposal, *, dependencies: list[Path] | None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(proposal.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    register_artifact(root, path, dependencies=dependencies)
