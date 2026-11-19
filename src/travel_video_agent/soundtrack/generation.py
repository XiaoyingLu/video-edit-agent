"""Project-level Stable Audio generation workflow and provenance artifacts."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from ..artifact_tracking import register_artifact
from .providers.stable_audio import (
    StableAudioGenerationProvider,
    StableAudioGenerationRequest,
    StableAudioGenerationResult,
)
from .schemas import SoundtrackBrief, SoundtrackLicense, SoundtrackLicenseType


class SoundtrackGenerationError(ValueError):
    """Raised when a project cannot safely generate or persist a soundtrack."""


class GeneratedSoundtrackArtifact(BaseModel):
    """Persisted generation result, rights evidence, and project provenance."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    project_path: str
    brief_path: str
    audio_path: str
    provider_result: StableAudioGenerationResult
    license: SoundtrackLicense
    generated_at: datetime
    human_review_required: bool = True
    selected: bool = False
    timeline_modified: bool = False
    rendering_performed: bool = False


def _project_path(root: Path, value: Path, label: str) -> Path:
    candidate = value.expanduser()
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SoundtrackGenerationError(f"{label} must be inside the project folder") from exc
    return resolved


def stable_audio_license() -> SoundtrackLicense:
    """Return the published Stable Audio API permissions, with their scope recorded."""
    return SoundtrackLicense(
        license_type=SoundtrackLicenseType.CUSTOM,
        license_name="Stability AI Stable Audio API commercial-use terms",
        license_url=(
            "https://kb.stability.ai/knowledge-base/"
            "stable-audio-commerical-and-usage-licensing"
        ),
        evidence_url="https://docs.stability.ai/api-reference/stable-audio/text-to-audio",
        commercial_use_allowed=True,
        monetized_video_allowed=True,
        derivatives_allowed=True,
        synchronization_allowed=True,
        attribution_required=False,
        terms_snapshot=(
            "Stability AI's Stable Audio licensing guide explicitly permits commercial use, "
            "including monetized YouTube and social-media videos. API-generated audio is "
            "subject to the current Terms of Service and Acceptable Use Policy. The user "
            "remains responsible for reviewing output for third-party rights."
        ),
        account_constraints=[
            "Use is subject to the current Stability AI Terms of Service "
            "and Acceptable Use Policy.",
            "Review generated output for third-party intellectual-property "
            "conflicts before publication.",
            "Commercial use is permitted by the published Stable Audio API licensing guide.",
        ],
    )


async def generate_soundtrack_from_brief(
    folder: Path,
    *,
    output_path: Path | None = None,
    output_format: str = "mp3",
    seed: int | None = None,
    timeout_seconds: float = 900,
    poll_interval_seconds: float = 10,
) -> tuple[GeneratedSoundtrackArtifact, Path]:
    """Generate audio from the project's brief and track both audio and metadata."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project folder does not exist: {root}")
    brief_file = root / ".video-agent" / "soundtrack-brief.json"
    if not brief_file.is_file():
        raise FileNotFoundError(f"Missing soundtrack brief: {brief_file}")
    brief = SoundtrackBrief.model_validate_json(brief_file.read_text(encoding="utf-8"))
    if Path(brief.project_path).resolve() != root:
        raise SoundtrackGenerationError("Soundtrack brief belongs to a different project")
    if brief.target_duration_seconds > 380:
        raise SoundtrackGenerationError(
            "Stable Audio 3.0 supports up to 380 seconds per generation; "
            "shorten the soundtrack brief before generating."
        )
    if output_format not in {"mp3", "wav"}:
        raise SoundtrackGenerationError("Output format must be mp3 or wav")
    destination = output_path or Path(
        f".video-agent/generated-soundtracks/track-{uuid4().hex[:12]}.{output_format}"
    )
    destination = _project_path(root, destination, "Generated audio output")
    if destination.suffix.lower() != f".{output_format}":
        raise SoundtrackGenerationError(
            f"Output path must use the .{output_format} extension"
        )
    metadata_path = destination.with_suffix(".json")
    if metadata_path.exists():
        raise FileExistsError(f"Generation metadata already exists: {metadata_path}")
    request = StableAudioGenerationRequest(
        prompt=brief.generation_prompt,
        duration_seconds=brief.target_duration_seconds,
        output_format=output_format,
        seed=seed,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )
    provider = StableAudioGenerationProvider()
    result = await provider.generate(request, output_path=destination)
    artifact = GeneratedSoundtrackArtifact(
        project_path=str(root),
        brief_path=brief_file.relative_to(root).as_posix(),
        audio_path=destination.relative_to(root).as_posix(),
        provider_result=result,
        license=stable_audio_license(),
        generated_at=datetime.now(UTC),
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    register_artifact(root, destination, dependencies=[brief_file])
    register_artifact(root, metadata_path, dependencies=[brief_file, destination])
    return artifact, metadata_path
