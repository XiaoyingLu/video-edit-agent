"""Build a deterministic soundtrack brief from an approved creative plan."""

import json
from pathlib import Path

from ..artifact_tracking import register_artifact, validate_artifact
from ..editing.planning_schemas import VideoProjectPlan
from .schemas import SoundtrackBrief, SoundtrackEnergySegment


_STYLE_DIRECTIONS = {
    "cinematic": (
        ["cinematic", "ambient", "post-rock", "orchestral"],
        ["warm piano", "soft strings", "atmospheric pads", "subtle percussion"],
    ),
    "vlog": (
        ["indie electronic", "acoustic", "chill pop"],
        ["acoustic guitar", "light percussion", "warm bass"],
    ),
    "documentary": (
        ["ambient", "minimal cinematic", "organic instrumental"],
        ["piano", "subtle strings", "natural textures"],
    ),
    "highlight_reel": (
        ["uplifting electronic", "indie pop", "cinematic percussion"],
        ["driving percussion", "synth pulses", "bright piano"],
    ),
    "social_media": (
        ["upbeat electronic", "indie pop", "future bass"],
        ["punchy drums", "bright synths", "bass"],
    ),
}

_MOOD_TERMS = {
    "wonder": ("wonder", "awe", "discovery", "majestic", "breathtaking"),
    "calm": ("calm", "peace", "serene", "quiet", "gentle", "reflection"),
    "adventurous": ("adventure", "explore", "journey", "hike", "action", "thrill"),
    "uplifting": ("joy", "celebration", "uplifting", "memorable", "inspiring"),
    "nostalgic": ("nostalgia", "memory", "reminisce", "sentimental"),
    "mysterious": ("mystery", "unknown", "anticipation", "dramatic"),
}


def _infer_moods(plan: VideoProjectPlan) -> list[str]:
    text = " ".join(
        [
            plan.brief.source_prompt,
            plan.story_treatment.logline if plan.story_treatment else "",
            plan.story_treatment.emotional_journey if plan.story_treatment else "",
            *(scene.emotional_intent for scene in plan.scenes),
            *(scene.purpose for scene in plan.scenes),
        ]
    ).lower()
    matches = [
        mood for mood, terms in _MOOD_TERMS.items() if any(term in text for term in terms)
    ]
    if not matches:
        matches = ["cinematic", "uplifting"] if plan.brief.style != "documentary" else ["reflective"]
    return matches[:4]


def _scene_energy(scene) -> str:
    text = f"{scene.narrative_function} {scene.emotional_intent} {scene.purpose}".lower()
    if scene.narrative_function in {"climax", "turning_point"} or any(
        word in text for word in ("action", "peak", "exciting", "climax", "energetic")
    ):
        return "high"
    if scene.narrative_function in {"breath", "resolution"} or any(
        word in text for word in ("quiet", "calm", "reflect", "peace", "gentle")
    ):
        return "low"
    return "medium"


def _tempo_range(style: str, pacing: str) -> tuple[int, int]:
    if pacing == "slow":
        return 65, 105
    if pacing == "fast":
        return 105, 145
    if style in {"cinematic", "documentary"}:
        return 75, 120
    return 90, 130


def build_soundtrack_brief(plan: VideoProjectPlan, source_plan_path: str) -> SoundtrackBrief:
    """Derive music direction from narrative, style, and ordered scene intent."""
    genres, instruments = _STYLE_DIRECTIONS[plan.brief.style]
    moods = _infer_moods(plan)
    summary = (
        plan.story_treatment.logline
        if plan.story_treatment
        else plan.brief.source_prompt
    )
    segments = []
    cursor = 0.0
    for scene in plan.scenes:
        end = cursor + scene.target_duration_seconds
        segments.append(
            SoundtrackEnergySegment(
                scene_id=scene.scene_id,
                scene_title=scene.title,
                start_seconds=round(cursor, 3),
                end_seconds=round(end, 3),
                energy=_scene_energy(scene),
                emotional_intent=scene.emotional_intent,
            )
        )
        cursor = end
    if not segments:
        segments = [
            SoundtrackEnergySegment(
                scene_id="full-video",
                scene_title="Full video",
                start_seconds=0,
                end_seconds=plan.brief.target_duration_seconds,
                energy="medium",
                emotional_intent=plan.story_treatment.emotional_journey
                if plan.story_treatment
                else "",
            )
        ]

    tempo_min, tempo_max = _tempo_range(plan.brief.style, plan.brief.pacing)
    search_terms = list(dict.fromkeys([*moods, *genres, plan.brief.style, "instrumental"]))
    generation_prompt = (
        f"Create an original {plan.brief.style} instrumental soundtrack, "
        f"{plan.brief.target_duration_seconds:.0f} seconds long. "
        f"Mood: {', '.join(moods)}. Musical direction: {', '.join(genres)}. "
        f"Instrumentation: {', '.join(instruments)}. "
        f"Tempo around {tempo_min}-{tempo_max} BPM. "
        "Follow the supplied narrative energy arc, with natural transitions and a resolved ending. "
        "Avoid vocals, recognizable melodies, and soundalikes of existing artists or songs."
    )
    return SoundtrackBrief(
        project_path=plan.project_path,
        source_video_plan=source_plan_path,
        target_duration_seconds=plan.brief.target_duration_seconds,
        video_style=plan.brief.style,
        narrative_summary=summary,
        mood=moods,
        genre_directions=genres,
        instrumentation=instruments,
        vocals="instrumental_preferred",
        tempo_min_bpm=tempo_min,
        tempo_max_bpm=tempo_max,
        energy_arc=segments,
        search_terms=search_terms[:12],
        generation_prompt=generation_prompt,
    )


def create_soundtrack_brief(
    folder: Path,
    *,
    plan_path: Path | None = None,
    output_path: Path | None = None,
) -> tuple[SoundtrackBrief, Path]:
    """Load a project plan, create a brief, and track its source dependency."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Project folder does not exist: {root}")
    source = (plan_path or root / ".video-agent" / "video-project-plan.json").expanduser().resolve()
    destination = (
        output_path or root / ".video-agent" / "soundtrack-brief.json"
    ).expanduser().resolve()
    for candidate, label in ((source, "video plan"), (destination, "soundtrack brief")):
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{label} must be inside the project folder") from exc
    if destination == source:
        raise ValueError("soundtrack brief output cannot overwrite the source video plan")
    if not source.is_file():
        raise FileNotFoundError(f"Video project plan does not exist: {source}")
    validate_artifact(root, source)
    plan = VideoProjectPlan.model_validate_json(source.read_text(encoding="utf-8"))
    if Path(plan.project_path).expanduser().resolve() != root:
        raise ValueError("video project plan belongs to a different project folder")

    brief = build_soundtrack_brief(plan, source.relative_to(root).as_posix())
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(brief.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    register_artifact(root, destination, dependencies=[source])
    return brief, destination
