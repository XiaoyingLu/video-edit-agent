"""Hierarchical creative planning artifacts, separate from executable edit timelines."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .intent_schemas import EditingPlan


class CreativeBrief(BaseModel):
    """User-facing creative intent and distribution constraints."""

    model_config = ConfigDict(extra="forbid")

    source_prompt: str = Field(min_length=1, max_length=4000)
    intended_audience: str | None = Field(default=None, max_length=160)
    platform: Literal["general", "youtube", "youtube_shorts", "instagram_reels", "tiktok"] = (
        "general"
    )
    target_duration_seconds: float = Field(gt=0, le=3600)
    style: Literal["cinematic", "vlog", "documentary", "highlight_reel", "social_media"]
    aspect_ratio: Literal["16:9", "9:16", "1:1"]
    resolution: Literal["720p", "1080p", "4k"]
    pacing: Literal["slow", "moderate", "fast"]
    creative_priorities: list[
        Literal[
            "scenic_landscapes",
            "city_highlights",
            "people_and_activities",
            "food_and_culture",
            "architecture",
            "memorable_moments",
            "chronological_story",
        ]
    ] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list, max_length=10)


class StoryTreatment(BaseModel):
    """Narrative concept developed from an approved creative brief."""

    model_config = ConfigDict(extra="forbid")

    working_title: str = Field(min_length=1, max_length=120)
    logline: str = Field(min_length=1, max_length=400)
    narrative_arc: list[str] = Field(min_length=2, max_length=8)
    emotional_journey: str = Field(min_length=1, max_length=300)


class StoryboardShot(BaseModel):
    """One intended visual beat; footage assignment is a later planning stage."""

    model_config = ConfigDict(extra="forbid")

    shot_id: str = Field(min_length=1, max_length=80)
    scene_id: str = Field(min_length=1, max_length=80)
    sequence_index: int = Field(ge=1)
    purpose: str = Field(min_length=1, max_length=240)
    visual_description: str = Field(min_length=1, max_length=600)
    shot_scale: Literal[
        "extreme_wide", "wide", "medium", "close_up", "extreme_close_up", "aerial", "unknown"
    ] = "unknown"
    visual_role: Literal[
        "establishing",
        "action",
        "detail",
        "reaction",
        "transition",
        "portrait",
        "scenic_breath",
        "unknown",
    ] = "unknown"
    target_duration_seconds: float = Field(gt=0, le=60)
    narration: str | None = Field(default=None, max_length=1000)
    on_screen_text: str | None = Field(default=None, max_length=240)
    audio_cue: str | None = Field(default=None, max_length=240)
    music_energy: Literal["low", "medium", "high", "unknown"] = "unknown"
    music_energy_sequence: list[Literal["low", "medium", "high"]] = Field(default_factory=list)
    music_section_start_seconds: float | None = Field(default=None, ge=0)
    music_section_end_seconds: float | None = Field(default=None, gt=0)
    music_alignment_note: str = Field(default="", max_length=240)
    pacing_recommendation: Literal["linger", "steady", "brisk", "unknown"] = "unknown"
    pacing_rationale: str = Field(default="", max_length=240)
    transition_recommendation: Literal[
        "soft_dissolve", "match_cut", "cut_on_action", "straight_cut", "end_on_hold", "unknown"
    ] = "unknown"
    transition_rationale: str = Field(default="", max_length=240)
    primary_subject: str = Field(default="", max_length=100)
    temporal_summary: str = Field(default="", max_length=300)
    action_progression: list[str] = Field(default_factory=list, max_length=6)
    dynamics_confidence: float = Field(default=0, ge=0, le=1)
    subject_motion: Literal[
        "approaching", "receding", "crossing", "stationary", "interacting", "mixed", "unknown"
    ] = "unknown"
    camera_motion: Literal[
        "static", "pan", "tilt", "zoom", "tracking", "handheld", "mixed", "unknown"
    ] = "unknown"
    composition: Literal[
        "centered",
        "rule_of_thirds",
        "leading_lines",
        "symmetrical",
        "layered_depth",
        "negative_space",
        "unknown",
    ] = "unknown"
    footage_path: str | None = None
    footage_start_seconds: float | None = Field(default=None, ge=0)
    footage_end_seconds: float | None = Field(default=None, gt=0)
    assignment_status: Literal["unassigned", "suggested", "approved"] = "unassigned"

    @model_validator(mode="after")
    def validate_footage_interval(self) -> "StoryboardShot":
        if (self.music_section_start_seconds is None) != (self.music_section_end_seconds is None):
            raise ValueError("music section start and end must be supplied together")
        if (
            self.music_section_start_seconds is not None
            and self.music_section_end_seconds is not None
            and self.music_section_end_seconds <= self.music_section_start_seconds
        ):
            raise ValueError("music section end must be greater than its start")
        if (self.footage_start_seconds is None) != (self.footage_end_seconds is None):
            raise ValueError("footage start and end must be supplied together")
        if (
            self.footage_start_seconds is not None
            and self.footage_end_seconds is not None
            and self.footage_end_seconds <= self.footage_start_seconds
        ):
            raise ValueError("footage end must be greater than footage start")
        if self.assignment_status == "unassigned" and self.footage_path is not None:
            raise ValueError("unassigned shots cannot reference footage")
        if self.assignment_status != "unassigned" and self.footage_path is None:
            raise ValueError("assigned shots must reference a footage path")
        return self


class PlannedScene(BaseModel):
    """A narrative section containing an ordered set of storyboard shots."""

    model_config = ConfigDict(extra="forbid")

    scene_id: str = Field(min_length=1, max_length=80)
    sequence_index: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=120)
    purpose: str = Field(min_length=1, max_length=300)
    narrative_function: Literal[
        "hook", "establishing", "development", "turning_point", "climax", "resolution", "breath"
    ]
    target_duration_seconds: float = Field(gt=0, le=3600)
    emotional_intent: str = Field(default="", max_length=200)
    candidate_scene_ids: list[str] = Field(default_factory=list, max_length=40)
    must_include_scene_ids: list[str] = Field(default_factory=list, max_length=10)
    music_energy: Literal["low", "medium", "high", "unknown"] = "unknown"
    music_energy_sequence: list[Literal["low", "medium", "high"]] = Field(default_factory=list)
    music_section_start_seconds: float | None = Field(default=None, ge=0)
    music_section_end_seconds: float | None = Field(default=None, gt=0)
    music_alignment_note: str = Field(default="", max_length=240)
    shots: list[StoryboardShot] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_music_section(self) -> "PlannedScene":
        if (self.music_section_start_seconds is None) != (self.music_section_end_seconds is None):
            raise ValueError("music section start and end must be supplied together")
        if (
            self.music_section_start_seconds is not None
            and self.music_section_end_seconds is not None
            and self.music_section_end_seconds <= self.music_section_start_seconds
        ):
            raise ValueError("music section end must be greater than its start")
        return self


class VideoProjectPlan(BaseModel):
    """Top-level plan artifact; intentionally distinct from a renderer timeline."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    project_path: str
    source_editing_plan: str = ".video-agent/editing-plan.json"
    source_music_analysis: str | None = None
    brief: CreativeBrief
    editing_preferences: EditingPlan
    story_treatment: StoryTreatment | None = None
    scenes: list[PlannedScene] = Field(default_factory=list)
    planning_stage: Literal[
        "brief_ready", "story_ready", "storyboard_ready", "footage_mapped", "production_ready"
    ] = "brief_ready"
    approval_required: Literal[True] = True
    approved: bool = False
    rendering_performed: Literal[False] = False

    @model_validator(mode="after")
    def validate_plan(self) -> "VideoProjectPlan":
        scene_ids = [scene.scene_id for scene in self.scenes]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("scene IDs must be unique")
        if [scene.sequence_index for scene in self.scenes] != list(range(1, len(self.scenes) + 1)):
            raise ValueError("scene sequence indices must be contiguous and ordered from 1")
        shot_ids = [shot.shot_id for scene in self.scenes for shot in scene.shots]
        if len(shot_ids) != len(set(shot_ids)):
            raise ValueError("shot IDs must be unique across the storyboard")
        if any(shot.scene_id != scene.scene_id for scene in self.scenes for shot in scene.shots):
            raise ValueError("each storyboard shot must reference its containing scene")
        if self.planning_stage == "brief_ready" and (
            self.story_treatment is not None or self.scenes
        ):
            raise ValueError("brief_ready plans cannot contain story or storyboard content")
        if (
            self.planning_stage
            in {"story_ready", "storyboard_ready", "footage_mapped", "production_ready"}
            and self.story_treatment is None
        ):
            raise ValueError("story stages require a story treatment")
        if self.planning_stage in {"storyboard_ready", "footage_mapped", "production_ready"}:
            if not self.scenes:
                raise ValueError("storyboard stages require at least one scene")
        if self.planning_stage == "story_ready" and self.scenes:
            raise ValueError("story_ready plans cannot contain storyboard scenes")
        if self.planning_stage in {"footage_mapped", "production_ready"}:
            shots = [shot for scene in self.scenes for shot in scene.shots]
            if not shots or any(shot.assignment_status == "unassigned" for shot in shots):
                raise ValueError("footage-mapped stages require every shot to have an assignment")
        if self.planning_stage in {"storyboard_ready", "footage_mapped", "production_ready"}:
            if any(not scene.shots for scene in self.scenes):
                raise ValueError("every storyboard scene must contain at least one shot")
            for scene in self.scenes:
                shot_duration = sum(shot.target_duration_seconds for shot in scene.shots)
                if abs(scene.target_duration_seconds - shot_duration) > 0.05:
                    raise ValueError("scene duration must equal the sum of its shot durations")
            storyboard_duration = sum(
                shot.target_duration_seconds for scene in self.scenes for shot in scene.shots
            )
            if abs(storyboard_duration - self.brief.target_duration_seconds) > 0.05:
                raise ValueError(
                    "storyboard duration must equal the creative brief target duration"
                )
        if self.planning_stage == "production_ready":
            shots = [shot for scene in self.scenes for shot in scene.shots]
            if not self.approved or any(shot.assignment_status != "approved" for shot in shots):
                raise ValueError("production-ready plans require approved footage assignments")
        return self
