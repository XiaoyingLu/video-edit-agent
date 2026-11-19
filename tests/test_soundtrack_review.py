import json
from datetime import UTC, datetime

import pytest

from travel_video_agent.soundtrack.catalog import SoundtrackSearchResponse
from travel_video_agent.soundtrack.review import (
    SoundtrackReviewError,
    build_soundtrack_candidate_review,
    save_soundtrack_candidate_review,
    select_soundtrack_candidate,
)
from travel_video_agent.soundtrack.schemas import (
    SoundtrackBrief,
    SoundtrackCandidate,
    SoundtrackLicense,
)


def _brief(root):
    return SoundtrackBrief(
        project_path=str(root.resolve()),
        source_video_plan=".video-agent/video-project-plan.json",
        target_duration_seconds=60,
        video_style="cinematic",
        narrative_summary="A quiet mountain journey ending with a summit reveal.",
        mood=["calm", "wonder"],
        genre_directions=["ambient", "cinematic"],
        instrumentation=["piano"],
        vocals="instrumental_preferred",
        tempo_min_bpm=70,
        tempo_max_bpm=110,
        energy_arc=[
            {
                "scene_id": "scene-1",
                "scene_title": "Journey",
                "start_seconds": 0,
                "end_seconds": 60,
                "energy": "medium",
            }
        ],
        search_terms=["calm", "ambient"],
        generation_prompt="A calm cinematic instrumental.",
    )


def _candidate(candidate_id="track-1", *, eligible=True):
    return SoundtrackCandidate(
        candidate_id=candidate_id,
        origin="catalog",
        provider_id="test",
        provider_asset_id=candidate_id,
        title='<Mountain "Light">',
        creator="Example Artist",
        duration_seconds=60,
        tempo_bpm=90,
        instrumental=True,
        tags=["calm", "wonder", "ambient", "cinematic", "piano"],
        source_url="https://example.com/track",
        preview_url="https://cdn.example.com/preview.mp3",
        retrieved_at=datetime.now(UTC),
        license=SoundtrackLicense(
            license_type="custom" if eligible else "unknown",
            license_name="Commercial API agreement" if eligible else None,
            license_url="https://example.com/license" if eligible else None,
            evidence_url="https://example.com/terms" if eligible else None,
            commercial_use_allowed=True if eligible else None,
            monetized_video_allowed=True if eligible else None,
            derivatives_allowed=True if eligible else None,
            synchronization_allowed=True if eligible else None,
        ),
    )


def _write_inputs(root, candidate):
    artifact_dir = root / ".video-agent"
    artifact_dir.mkdir(exist_ok=True)
    brief_path = artifact_dir / "soundtrack-brief.json"
    brief_path.write_text(_brief(root).model_dump_json(), encoding="utf-8")
    candidates_path = artifact_dir / "soundtrack-candidates.json"
    candidates_path.write_text(
        SoundtrackSearchResponse(provider_id="test", candidates=[candidate]).model_dump_json(),
        encoding="utf-8",
    )
    return candidates_path


def test_review_builds_ranked_html_audition_with_optional_video(tmp_path):
    candidates_path = _write_inputs(tmp_path, _candidate())
    video = tmp_path / "draft.mp4"
    video.write_bytes(b"fake video")

    report = build_soundtrack_candidate_review(
        tmp_path, candidates_path=candidates_path, video_path=video
    )
    json_path, html_path = save_soundtrack_candidate_review(tmp_path, report)

    page = html_path.read_text(encoding="utf-8")
    assert "<video controls" in page
    assert '<audio controls' in page
    assert "&lt;Mountain &quot;Light&quot;&gt;" in page
    assert "Commercial API agreement" not in page
    assert report.items[0].ranking.rights.status == "eligible"
    assert json.loads(json_path.read_text(encoding="utf-8"))["timeline_modified"] is False


def test_select_requires_explicitly_eligible_candidate(tmp_path):
    candidates_path = _write_inputs(tmp_path, _candidate())
    report = build_soundtrack_candidate_review(tmp_path, candidates_path=candidates_path)
    save_soundtrack_candidate_review(tmp_path, report)

    selection, destination = select_soundtrack_candidate(
        tmp_path, candidate_id="track-1"
    )

    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert selection.explicit_human_selection is True
    assert selection.audio_downloaded is False
    assert selection.timeline_modified is False
    assert saved["candidate"]["candidate_id"] == "track-1"


def test_unknown_rights_candidate_cannot_be_selected(tmp_path):
    candidates_path = _write_inputs(tmp_path, _candidate(eligible=False))
    report = build_soundtrack_candidate_review(tmp_path, candidates_path=candidates_path)
    save_soundtrack_candidate_review(tmp_path, report)

    with pytest.raises(SoundtrackReviewError, match="cannot be selected"):
        select_soundtrack_candidate(tmp_path, candidate_id="track-1")


def test_review_rejects_paths_outside_project(tmp_path):
    candidates_path = _write_inputs(tmp_path, _candidate())
    with pytest.raises(SoundtrackReviewError, match="inside the project"):
        build_soundtrack_candidate_review(
            tmp_path, candidates_path=candidates_path, video_path=tmp_path.parent / "outside.mp4"
        )


def test_review_does_not_overwrite_existing_output(tmp_path):
    candidates_path = _write_inputs(tmp_path, _candidate())
    report = build_soundtrack_candidate_review(tmp_path, candidates_path=candidates_path)
    save_soundtrack_candidate_review(tmp_path, report)
    with pytest.raises(FileExistsError):
        save_soundtrack_candidate_review(tmp_path, report)



def test_soundstripe_search_uses_brief_and_tracks_license_profile(tmp_path, monkeypatch):
    import asyncio

    from travel_video_agent.soundtrack import review
    from travel_video_agent.soundtrack.schemas import SoundtrackLicense

    _write_inputs(tmp_path, _candidate())
    profile_path = tmp_path / ".video-agent" / "soundstripe-license.json"
    profile_path.write_text(
        SoundtrackLicense(
            license_type="custom",
            license_name="Verified account agreement",
            license_url="https://example.com/license",
            evidence_url="https://example.com/terms",
            commercial_use_allowed=True,
            monetized_video_allowed=True,
            derivatives_allowed=True,
            synchronization_allowed=True,
        ).model_dump_json(),
        encoding="utf-8",
    )
    expected = SoundtrackSearchResponse(provider_id="soundstripe", candidates=[])

    class FakeProvider:
        def __init__(self, *, license_profile=None):
            assert license_profile.license_name == "Verified account agreement"

        async def search(self, request):
            assert request.target_duration_seconds == 60
            assert request.moods == ["calm", "wonder"]
            assert request.tempo_min_bpm == 70
            return expected

    monkeypatch.setattr(review, "SoundstripeCatalogProvider", FakeProvider)
    response, destination = asyncio.run(
        review.search_soundstripe_catalog(
            tmp_path, license_profile_path=profile_path
        )
    )

    assert response == expected
    assert destination == tmp_path / ".video-agent" / "soundtrack-candidates.json"
    assert json.loads(destination.read_text(encoding="utf-8")) == expected.model_dump(mode="json")
