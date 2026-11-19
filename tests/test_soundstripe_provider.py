import pytest

from travel_video_agent.soundtrack.catalog import SoundtrackSearchRequest
from travel_video_agent.soundtrack.providers.soundstripe import SoundstripeCatalogProvider
from travel_video_agent.soundtrack.schemas import SoundtrackLicense


def provider(profile=None):
    return SoundstripeCatalogProvider(api_key="test-key", license_profile=profile)


def payload():
    return {
        "data": [
            {
                "id": "track-1",
                "type": "songs",
                "attributes": {
                    "title": "Mountain Light",
                    "bpm": 92,
                    "tags": {"mood": ["calm"], "genre": ["ambient"], "instrument": ["piano"]},
                },
                "relationships": {
                    "artists": {"data": [{"id": "artist-1", "type": "artists"}]},
                    "audio_files": {"data": [{"id": "audio-1", "type": "audio_files"}]},
                },
            }
        ],
        "included": [
            {"id": "artist-1", "type": "artists", "attributes": {"name": "Example Artist"}},
            {
                "id": "audio-1",
                "type": "audio_files",
                "attributes": {
                    "duration": 101.5,
                    "instrumental": True,
                    "versions": {"mp3": "https://cdn.example.com/preview.mp3"},
                },
            },
        ],
        "links": {"next": "https://api.soundstripe.com/v1/songs?page%5Bnumber%5D=2"},
    }


def test_api_key_is_required(monkeypatch):
    monkeypatch.delenv("SOUNDSTRIPE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SOUNDSTRIPE_API_KEY"):
        SoundstripeCatalogProvider(api_key="")


def test_search_filters_and_pagination_are_mapped():
    request = SoundtrackSearchRequest(
        query="cinematic trip",
        moods=["calm"],
        genres=["ambient"],
        search_terms=["piano"],
        target_duration_seconds=100,
        duration_tolerance_seconds=15,
        tempo_min_bpm=80,
        tempo_max_bpm=110,
        limit=12,
        cursor="3",
    )
    params = provider()._build_params(request)
    assert params["filter[q]"] == "cinematic trip piano"
    assert params["filter[duration][min]"] == "85"
    assert params["filter[duration][max]"] == "115"
    assert params["filter[bpm][min]"] == "80"
    assert params["filter[bpm][max]"] == "110"
    assert params["filter[instrumental]"] == "true"
    assert params["filter[tags][mood]"] == "calm"
    assert params["filter[tags][genre]"] == "ambient"
    assert "filter[tags][instrument]" not in params
    assert params["page[number]"] == "3"


def test_response_normalizes_song_audio_and_artist_resources():
    response = provider()._normalize_response(payload())
    assert response.provider_id == "soundstripe"
    assert response.next_cursor == "2"
    assert len(response.candidates) == 1
    candidate = response.candidates[0]
    assert candidate.candidate_id == "soundstripe:track-1"
    assert candidate.title == "Mountain Light"
    assert candidate.creator == "Example Artist"
    assert candidate.duration_seconds == 101.5
    assert candidate.tempo_bpm == 92
    assert candidate.instrumental is True
    assert candidate.tags == ["ambient", "calm", "piano"]
    assert str(candidate.preview_url) == "https://cdn.example.com/preview.mp3"
    assert candidate.license.license_type == "unknown"


def test_explicit_account_license_profile_is_copied():
    profile = SoundtrackLicense(
        license_type="custom",
        license_name="Account API agreement",
        license_url="https://www.soundstripe.com/api",
        evidence_url="https://www.soundstripe.com/api",
        commercial_use_allowed=True,
        monetized_video_allowed=True,
        derivatives_allowed=True,
        synchronization_allowed=True,
        terms_snapshot="Confirmed by account owner.",
    )
    candidate = provider(profile)._normalize_response(payload()).candidates[0]
    assert candidate.license.commercial_use_allowed is True
    assert candidate.license.terms_snapshot == "Confirmed by account owner."
    assert candidate.license is not profile


def test_missing_duration_is_not_guessed():
    data = payload()
    data["included"][1]["attributes"].pop("duration")
    assert provider()._normalize_response(data).candidates == []


def test_invalid_cursor_is_rejected():
    request = SoundtrackSearchRequest(query="mountain", target_duration_seconds=100, cursor="next")
    with pytest.raises(ValueError, match="page number"):
        provider()._build_params(request)
