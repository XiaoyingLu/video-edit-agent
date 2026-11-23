"""Analyze the explicitly selected soundtrack and retain its licensing provenance."""

from ..atomic_io import write_json_atomic
import hashlib
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..artifact_tracking import register_artifact, validate_artifact
from ..music.pipeline import analyze_music
from ..music.schemas import MusicAnalysis
from .review import SoundtrackCandidateReview, SoundtrackSelection


class SelectedSoundtrackAnalysisError(ValueError):
    """Raised when the selected soundtrack cannot be safely analyzed."""


class SelectedSoundtrackAnalysisRecord(BaseModel):
    """Traceable link between a human selection, local audio, and music analysis."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    project_path: str
    selection_path: str
    review_path: str
    analysis_path: str
    candidate_id: str
    provider_id: str
    origin: str
    music_path: str
    audio_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    analyzed_at: datetime
    rights_status: str = "eligible"
    timeline_modified: bool = False
    rendering_performed: bool = False


def _project_file(root: Path, value: Path, label: str) -> Path:
    candidate = value.expanduser()
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SelectedSoundtrackAnalysisError(f"{label} must be inside the project folder") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def analyze_selected_soundtrack(
    folder: Path,
    *,
    selection_path: Path | None = None,
    music_path: Path | None = None,
    analysis_output_path: Path | None = None,
    record_output_path: Path | None = None,
) -> tuple[MusicAnalysis, SelectedSoundtrackAnalysisRecord, Path, Path]:
    """Analyze the selected local audio; catalog audio must be supplied explicitly."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project folder does not exist: {root}")
    selection_file = _project_file(
        root,
        selection_path or Path(".video-agent/soundtrack-selection.json"),
        "soundtrack selection",
    )
    if not validate_artifact(root, selection_file):
        raise SelectedSoundtrackAnalysisError(
            "Soundtrack selection is untracked or stale; recreate the selection before analysis"
        )
    selection = SoundtrackSelection.model_validate_json(selection_file.read_text(encoding="utf-8"))
    if Path(selection.project_path).resolve() != root:
        raise SelectedSoundtrackAnalysisError("Soundtrack selection belongs to another project")
    if not selection.explicit_human_selection:
        raise SelectedSoundtrackAnalysisError("An explicit human selection is required")
    if selection.ranking.rights.status != "eligible":
        raise SelectedSoundtrackAnalysisError("Selected soundtrack is not rights-eligible")

    review_file = _project_file(root, Path(selection.review_path), "soundtrack review")
    if not validate_artifact(root, review_file):
        raise SelectedSoundtrackAnalysisError(
            "Soundtrack review is untracked or stale; recreate the review before analysis"
        )
    report = SoundtrackCandidateReview.model_validate_json(review_file.read_text(encoding="utf-8"))
    if Path(report.project_path).resolve() != root:
        raise SelectedSoundtrackAnalysisError("Soundtrack review belongs to another project")
    matches = [
        item
        for item in report.items
        if item.candidate.candidate_id == selection.candidate.candidate_id
    ]
    if len(matches) != 1 or matches[0].ranking.rights.status != "eligible":
        raise SelectedSoundtrackAnalysisError(
            "Selected candidate is missing from the current eligible review"
        )
    if selection.candidate.origin == "generated":
        if not selection.candidate.asset_path:
            raise SelectedSoundtrackAnalysisError(
                "Generated selection has no project-local audio asset"
            )
        selected_audio = _project_file(
            root, Path(selection.candidate.asset_path), "generated soundtrack audio"
        )
        if (
            music_path is not None
            and _project_file(root, music_path, "soundtrack audio") != selected_audio
        ):
            raise SelectedSoundtrackAnalysisError(
                "Generated selections must be analyzed from their recorded audio asset"
            )
        if not selection.candidate.asset_sha256:
            raise SelectedSoundtrackAnalysisError(
                "Generated soundtrack is missing its recorded checksum"
            )
        audio_digest = hashlib.sha256(selected_audio.read_bytes()).hexdigest()
        if audio_digest != selection.candidate.asset_sha256:
            raise SelectedSoundtrackAnalysisError("Generated soundtrack checksum mismatch")
    else:
        if music_path is None:
            raise SelectedSoundtrackAnalysisError(
                "Catalog selections require --music with the licensed audio file saved "
                "inside the project. Preview URLs are not downloadable render assets."
            )
        selected_audio = _project_file(root, music_path, "licensed catalog soundtrack audio")

    audio_is_tracked = validate_artifact(root, selected_audio)
    audio_digest = hashlib.sha256(selected_audio.read_bytes()).hexdigest()
    analysis_file = analysis_output_path or Path(".video-agent/music-analysis.json")
    analysis_file = analysis_file.expanduser()
    analysis_file = (
        (root / analysis_file).resolve()
        if not analysis_file.is_absolute()
        else analysis_file.resolve()
    )
    record_file = record_output_path or Path(".video-agent/selected-soundtrack-analysis.json")
    record_file = record_file.expanduser()
    record_file = (
        (root / record_file).resolve() if not record_file.is_absolute() else record_file.resolve()
    )
    for path, label in ((analysis_file, "analysis output"), (record_file, "provenance output")):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SelectedSoundtrackAnalysisError(
                f"{label} must be inside the project folder"
            ) from exc
        if path.exists():
            raise FileExistsError(f"{label.capitalize()} already exists: {path}")
    if analysis_file == record_file:
        raise SelectedSoundtrackAnalysisError(
            "Analysis and provenance outputs must be different files"
        )

    analysis = analyze_music(root, selected_audio)
    if hashlib.sha256(selected_audio.read_bytes()).hexdigest() != audio_digest:
        raise SelectedSoundtrackAnalysisError(
            "Soundtrack audio changed while it was being analyzed"
        )
    if not audio_is_tracked:
        # Imported catalog audio is a user-provided local asset; track its exact content.
        register_artifact(root, selected_audio, dependencies=[selection_file, review_file])
    record = SelectedSoundtrackAnalysisRecord(
        project_path=str(root),
        selection_path=selection_file.relative_to(root).as_posix(),
        review_path=review_file.relative_to(root).as_posix(),
        analysis_path=analysis_file.relative_to(root).as_posix(),
        candidate_id=selection.candidate.candidate_id,
        provider_id=selection.candidate.provider_id,
        origin=selection.candidate.origin,
        music_path=selected_audio.relative_to(root).as_posix(),
        audio_sha256=audio_digest,
        analyzed_at=datetime.now(UTC),
    )
    analysis_file.parent.mkdir(parents=True, exist_ok=True)
    record_file.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(analysis_file, analysis)
    write_json_atomic(record_file, record)
    register_artifact(
        root,
        analysis_file,
        dependencies=[selection_file, review_file, selected_audio],
    )
    register_artifact(
        root,
        record_file,
        dependencies=[selection_file, review_file, selected_audio, analysis_file],
    )
    return analysis, record, analysis_file, record_file
