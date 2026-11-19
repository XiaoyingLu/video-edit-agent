"""Human-reviewed soundtrack candidate preview and selection."""

import asyncio
from datetime import UTC, datetime
import html
import json
from pathlib import Path
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field

from ..artifact_tracking import register_artifact, validate_artifact
from .catalog import SoundtrackSearchRequest, SoundtrackSearchResponse
from .ranking import SoundtrackCandidateRanking, rank_soundtrack_candidates
from .providers import SoundstripeCatalogProvider
from .schemas import SoundtrackBrief, SoundtrackCandidate, SoundtrackLicense


class SoundtrackReviewError(ValueError):
    """Raised when soundtrack review artifacts are unsafe or inconsistent."""


class SoundtrackReviewItem(BaseModel):
    """Candidate metadata paired with its explainable fit and rights assessment."""

    model_config = ConfigDict(extra="forbid")

    candidate: SoundtrackCandidate
    ranking: SoundtrackCandidateRanking


class SoundtrackCandidateReview(BaseModel):
    """Versioned, read-only review of soundtrack candidates for one project."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    project_path: str
    brief_path: str
    candidates_path: str
    video_path: str | None = None
    target_duration_seconds: float = Field(gt=0)
    items: list[SoundtrackReviewItem] = Field(default_factory=list)
    review_only: bool = True
    timeline_modified: bool = False
    selection_made: bool = False


class SoundtrackSelection(BaseModel):
    """Explicitly selected, rights-eligible soundtrack candidate."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    project_path: str
    review_path: str
    candidate: SoundtrackCandidate
    ranking: SoundtrackCandidateRanking
    selected_at: datetime
    explicit_human_selection: bool = True
    audio_downloaded: bool = False
    timeline_modified: bool = False
    rendering_performed: bool = False


def _project_file(root: Path, value: Path, label: str) -> Path:
    candidate = value.expanduser()
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SoundtrackReviewError(f"{label} must be inside the project folder") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved



async def search_soundstripe_catalog(
    folder: Path,
    *,
    license_profile_path: Path | None = None,
    output_path: Path | None = None,
    limit: int = 20,
) -> tuple[SoundtrackSearchResponse, Path]:
    """Search Soundstripe from the saved brief and persist normalized candidates."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project folder does not exist: {root}")
    brief_file = _project_file(
        root, Path(".video-agent/soundtrack-brief.json"), "soundtrack brief"
    )
    brief = SoundtrackBrief.model_validate_json(brief_file.read_text(encoding="utf-8"))
    if Path(brief.project_path).resolve() != root:
        raise SoundtrackReviewError("Soundtrack brief belongs to a different project")
    license_profile = None
    dependencies: list[Path] = [brief_file]
    if license_profile_path:
        profile_file = _project_file(root, license_profile_path, "Soundstripe license profile")
        license_profile = SoundtrackLicense.model_validate_json(
            profile_file.read_text(encoding="utf-8")
        )
        dependencies.append(profile_file)
    request = SoundtrackSearchRequest(
        query=" ".join(brief.search_terms[:3]) or brief.narrative_summary[:300],
        search_terms=brief.search_terms,
        moods=brief.mood,
        genres=brief.genre_directions,
        target_duration_seconds=brief.target_duration_seconds,
        tempo_min_bpm=brief.tempo_min_bpm,
        tempo_max_bpm=brief.tempo_max_bpm,
        instrumental_preferred=brief.vocals == "instrumental_preferred",
        limit=limit,
    )
    provider = SoundstripeCatalogProvider(license_profile=license_profile)
    response = await provider.search(request)
    destination = output_path or Path(".video-agent/soundtrack-candidates.json")
    destination = destination.expanduser()
    destination = (root / destination).resolve() if not destination.is_absolute() else destination.resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise SoundtrackReviewError("Candidate output must be inside the project folder") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(response.model_dump_json(indent=2) + "\n", encoding="utf-8")
    register_artifact(root, destination, dependencies=dependencies)
    return response, destination


def build_soundtrack_candidate_review(
    folder: Path,
    *,
    candidates_path: Path,
    brief_path: Path | None = None,
    video_path: Path | None = None,
) -> SoundtrackCandidateReview:
    """Rank a normalized catalog response and prepare safe preview metadata."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project folder does not exist: {root}")
    brief_file = _project_file(
        root, brief_path or Path(".video-agent/soundtrack-brief.json"), "soundtrack brief"
    )
    candidates_file = _project_file(root, candidates_path, "soundtrack candidates")
    video_file = _project_file(root, video_path, "video preview") if video_path else None
    brief = SoundtrackBrief.model_validate_json(brief_file.read_text(encoding="utf-8"))
    if Path(brief.project_path).resolve() != root:
        raise SoundtrackReviewError("Soundtrack brief belongs to a different project")
    response = SoundtrackSearchResponse.model_validate_json(
        candidates_file.read_text(encoding="utf-8")
    )
    candidate_ids = [item.candidate_id for item in response.candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise SoundtrackReviewError("Candidate IDs must be unique")
    rankings = rank_soundtrack_candidates(brief, response.candidates)
    ranking_by_id = {item.candidate_id: item for item in rankings}
    items = [
        SoundtrackReviewItem(candidate=candidate, ranking=ranking_by_id[candidate.candidate_id])
        for candidate in response.candidates
    ]
    items.sort(
        key=lambda item: (
            {"eligible": 0, "review_required": 1, "ineligible": 2}[
                item.ranking.rights.status
            ],
            -item.ranking.creative_score,
            item.candidate.candidate_id,
        )
    )
    return SoundtrackCandidateReview(
        project_path=str(root),
        brief_path=brief_file.relative_to(root).as_posix(),
        candidates_path=candidates_file.relative_to(root).as_posix(),
        video_path=video_file.relative_to(root).as_posix() if video_file else None,
        target_duration_seconds=brief.target_duration_seconds,
        items=items,
    )


def render_soundtrack_review_html(
    report: SoundtrackCandidateReview, *, video_source: str | None = None
) -> str:
    """Render a self-contained, read-only candidate audition page."""
    video_html = (
        f'<video controls preload="metadata" src="{html.escape(video_source, quote=True)}">'
        "</video>"
        if video_source
        else "<p>No video supplied. Use --video to compare tracks alongside a project-local video.</p>"
    )
    cards = []
    for index, item in enumerate(report.items, start=1):
        candidate, ranking = item.candidate, item.ranking
        preview = (
            f'<audio controls preload="none" src="{html.escape(str(candidate.preview_url), quote=True)}">'
            "</audio>"
            if candidate.preview_url and str(candidate.preview_url).startswith("https://")
            else "<p>Preview URL unavailable.</p>"
        )
        rights_reasons = "".join(
            f"<li>{html.escape(reason)}</li>" for reason in ranking.rights.reasons
        )
        tags = ", ".join(candidate.tags) or "No tags"
        cards.append(
            f'<section class="candidate"><h2>{index}. {html.escape(candidate.title)}</h2>'
            f'<p>{html.escape(candidate.creator or "Unknown creator")} · '
            f'{candidate.duration_seconds:.1f}s · '
            f'{candidate.tempo_bpm if candidate.tempo_bpm is not None else "—"} BPM</p>'
            f'<p>Creative fit: {ranking.creative_score:.0%} · '
            f'Rights: <strong>{html.escape(ranking.rights.status)}</strong></p>'
            f'<p>Tags: {html.escape(tags)}</p>{preview}'
            f'<p><a href="{html.escape(str(candidate.source_url), quote=True)}" '
            'target="_blank" rel="noopener noreferrer">Provider track details</a></p>'
            f'<ul>{rights_reasons}</ul></section>'
        )
    video_note = (
        "Play the video and one candidate preview to audition them together. "
        "Playback is manual and does not alter the video."
        if video_source
        else "Candidate previews are independent audio players."
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Soundtrack Candidate Review</title>
<style>
:root {{ color-scheme:light dark; font-family:system-ui,sans-serif; }}
body {{ max-width:1000px; margin:2rem auto; padding:0 1rem; }}
video {{ width:100%; max-height:560px; background:#111; }}
audio {{ width:min(100%,650px); }}
.candidate {{ border:1px solid #8886; border-radius:10px; padding:1rem; margin:1rem 0; }}
strong {{ text-transform:capitalize; }}
.note {{ color:GrayText; }}
</style></head><body>
<h1>Soundtrack Candidate Review</h1>
<p>Target duration: {report.target_duration_seconds:.1f}s ·
{len(report.items)} candidate(s)</p>
{video_html}
<p class="note">{html.escape(video_note)}</p>
{"".join(cards)}
<p class="note">Read-only review. No track was selected, downloaded, or added to the timeline.
Select an eligible candidate explicitly with the <code>select-soundtrack</code> CLI command.</p>
</body></html>"""


def save_soundtrack_candidate_review(
    folder: Path,
    report: SoundtrackCandidateReview,
    *,
    output_path: Path | None = None,
) -> tuple[Path, Path]:
    """Persist review JSON and HTML inside the project without overwriting prior output."""
    root = folder.expanduser().resolve()
    base = output_path or Path(".video-agent/soundtrack-review")
    base = base.expanduser()
    base = (root / base).resolve() if not base.is_absolute() else base.resolve()
    try:
        base.relative_to(root)
    except ValueError as exc:
        raise SoundtrackReviewError("Review output must be inside the project folder") from exc
    json_path, html_path = base.with_suffix(".json"), base.with_suffix(".html")
    if json_path.exists() or html_path.exists():
        raise FileExistsError(f"Soundtrack review already exists: {base}")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    video_source = None
    if report.video_path:
        video_source = quote(os.path.relpath(root / report.video_path, html_path.parent).replace(os.sep, "/"), safe="/:@-._~")
    html_path.write_text(
        render_soundtrack_review_html(report, video_source=video_source), encoding="utf-8"
    )
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    register_artifact(
        root,
        json_path,
        dependencies=[root / report.brief_path, root / report.candidates_path],
    )
    return json_path, html_path


def select_soundtrack_candidate(
    folder: Path,
    *,
    review_path: Path | None = None,
    candidate_id: str,
    output_path: Path | None = None,
) -> tuple[SoundtrackSelection, Path]:
    """Persist an explicit selection only when the candidate passes the rights gate."""
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Project folder does not exist: {root}")
    review_file = _project_file(
        root, review_path or Path(".video-agent/soundtrack-review.json"), "soundtrack review"
    )
    if not validate_artifact(root, review_file):
        raise SoundtrackReviewError("Soundtrack review is untracked; regenerate it before selection")
    report = SoundtrackCandidateReview.model_validate_json(
        review_file.read_text(encoding="utf-8")
    )
    if Path(report.project_path).resolve() != root:
        raise SoundtrackReviewError("Soundtrack review belongs to a different project")
    matches = [item for item in report.items if item.candidate.candidate_id == candidate_id]
    if not matches:
        raise SoundtrackReviewError(f"Unknown soundtrack candidate: {candidate_id}")
    item = matches[0]
    if item.ranking.rights.status != "eligible":
        raise SoundtrackReviewError(
            f"Candidate '{candidate_id}' cannot be selected: rights status is "
            f"{item.ranking.rights.status}"
        )
    destination = output_path or Path(".video-agent/soundtrack-selection.json")
    destination = destination.expanduser()
    destination = (root / destination).resolve() if not destination.is_absolute() else destination.resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise SoundtrackReviewError("Selection output must be inside the project folder") from exc
    if destination.exists():
        raise FileExistsError(f"Soundtrack selection already exists: {destination}")
    selection = SoundtrackSelection(
        project_path=str(root),
        review_path=review_file.relative_to(root).as_posix(),
        candidate=item.candidate,
        ranking=item.ranking,
        selected_at=datetime.now(UTC),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(selection.model_dump_json(indent=2) + "\n", encoding="utf-8")
    register_artifact(root, destination, dependencies=[review_file])
    return selection, destination
