"""Normalize generated audio into the shared soundtrack review workflow."""

from ..atomic_io import write_json_atomic
import hashlib
from pathlib import Path

from ..artifact_tracking import register_artifact, validate_artifact
from ..music import analyze_music
from ..music.pipeline import MusicAnalysisError
from .catalog import SoundtrackSearchResponse
from .generation import GeneratedSoundtrackArtifact
from .schemas import (
    SoundtrackBrief,
    SoundtrackCandidate,
    SoundtrackCandidateCollection,
)


class GeneratedCandidateError(ValueError):
    """Raised when generated audio cannot be safely evaluated."""


def _project_file(root: Path, value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise GeneratedCandidateError(f"{label} must be inside the project folder") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def prepare_generated_soundtrack_candidates(
    folder: Path,
    *,
    catalog_candidates_path: Path | None = None,
    output_path: Path | None = None,
) -> tuple[SoundtrackCandidateCollection, Path]:
    """Analyze generated tracks and combine them with optional catalog candidates."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project folder does not exist: {root}")
    brief_file = _project_file(root, ".video-agent/soundtrack-brief.json", "soundtrack brief")
    brief = SoundtrackBrief.model_validate_json(brief_file.read_text(encoding="utf-8"))
    if Path(brief.project_path).resolve() != root:
        raise GeneratedCandidateError("Soundtrack brief belongs to a different project")

    catalog_file = None
    candidates: list[SoundtrackCandidate] = []
    dependencies = [brief_file]
    if catalog_candidates_path is None:
        default_catalog = root / ".video-agent/soundtrack-candidates.json"
        catalog_candidates_path = default_catalog if default_catalog.is_file() else None
    if catalog_candidates_path is not None:
        catalog_file = _project_file(root, catalog_candidates_path, "catalog candidates")
        catalog_response = SoundtrackSearchResponse.model_validate_json(
            catalog_file.read_text(encoding="utf-8")
        )
        candidates.extend(catalog_response.candidates)
        dependencies.append(catalog_file)

    generated_dir = root / ".video-agent" / "generated-soundtracks"
    if generated_dir.is_dir():
        for metadata_file in sorted(generated_dir.glob("*.json")):
            audio_candidates = [
                metadata_file.with_suffix(".mp3"),
                metadata_file.with_suffix(".wav"),
            ]
            audio_file = next((path for path in audio_candidates if path.is_file()), None)
            if audio_file is None:
                continue
            metadata_file = _project_file(root, metadata_file, "generated soundtrack metadata")
            audio_file = _project_file(root, audio_file, "generated soundtrack audio")
            artifact = GeneratedSoundtrackArtifact.model_validate_json(
                metadata_file.read_text(encoding="utf-8")
            )
            if Path(artifact.project_path).resolve() != root:
                raise GeneratedCandidateError(
                    f"Generated soundtrack metadata belongs to another "
                    f"project: {metadata_file.name}"
                )
            if (root / Path(artifact.brief_path)).resolve() != brief_file:
                raise GeneratedCandidateError(
                    f"Generated soundtrack was created from a different brief: {metadata_file.name}"
                )
            if (root / Path(artifact.audio_path)).resolve() != audio_file:
                raise GeneratedCandidateError(
                    f"Generated soundtrack metadata points to a different "
                    f"audio file: {metadata_file.name}"
                )
            digest = hashlib.sha256(audio_file.read_bytes()).hexdigest()
            if digest != artifact.provider_result.sha256:
                raise GeneratedCandidateError(
                    f"Generated soundtrack checksum mismatch: {audio_file.name}"
                )
            if not validate_artifact(root, metadata_file):
                raise GeneratedCandidateError(
                    f"Generated soundtrack metadata is not tracked: {metadata_file.name}"
                )
            if not validate_artifact(root, audio_file):
                raise GeneratedCandidateError(
                    f"Generated soundtrack audio is not tracked: {audio_file.name}"
                )
            try:
                analysis = analyze_music(root, audio_file)
            except (MusicAnalysisError, OSError, ValueError) as exc:
                raise GeneratedCandidateError(
                    f"Could not analyze generated soundtrack {audio_file.name}: {exc}"
                ) from exc
            bpm = round(analysis.bpm) if analysis.bpm is not None else None
            if bpm is not None and not 40 <= bpm <= 240:
                bpm = None
            candidates.append(
                SoundtrackCandidate(
                    candidate_id=f"stable-audio:{artifact.provider_result.generation_id}",
                    origin="generated",
                    provider_id="stability_stable_audio",
                    provider_asset_id=artifact.provider_result.generation_id,
                    title=f"Generated soundtrack {artifact.provider_result.generation_id[:8]}",
                    creator="Stability AI Stable Audio 3.0",
                    duration_seconds=analysis.duration_seconds,
                    tempo_bpm=bpm,
                    instrumental=None,
                    tags=[],
                    source_url="https://docs.stability.ai/api-reference/stable-audio/text-to-audio",
                    asset_path=audio_file.relative_to(root).as_posix(),
                    retrieved_at=artifact.generated_at,
                    asset_sha256=digest,
                    license=artifact.license,
                )
            )
            dependencies.extend([metadata_file, audio_file])

    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise GeneratedCandidateError(
            "Candidate IDs must be unique across catalog and generated tracks"
        )
    if not candidates:
        raise GeneratedCandidateError(
            "No soundtrack candidates found. Search the catalog or generate a track first."
        )
    destination = output_path or Path(".video-agent/soundtrack-candidates-combined.json")
    destination = destination.expanduser()
    destination = (
        (root / destination).resolve() if not destination.is_absolute() else destination.resolve()
    )
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise GeneratedCandidateError(
            "Combined candidate output must be inside the project folder"
        ) from exc
    if destination.exists():
        raise FileExistsError(f"Combined candidate file already exists: {destination}")
    collection = SoundtrackCandidateCollection(
        project_path=str(root),
        brief_path=brief_file.relative_to(root).as_posix(),
        catalog_candidates_path=catalog_file.relative_to(root).as_posix() if catalog_file else None,
        candidates=candidates,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(destination, collection)
    register_artifact(root, destination, dependencies=dependencies)
    return collection, destination
