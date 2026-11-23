"""Soundstripe catalog adapter using its documented REST API."""

import asyncio
import json
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

from ...config import get_setting

from ..catalog import SoundtrackSearchRequest, SoundtrackSearchResponse
from ..schemas import SoundtrackCandidate, SoundtrackLicense


class SoundstripeAPIError(RuntimeError):
    """Raised when Soundstripe rejects a catalog request or returns invalid data."""


class SoundstripeCatalogProvider:
    """Search Soundstripe songs; rights require an explicitly confirmed account profile."""

    provider_id = "soundstripe"
    api_base_url = "https://api.soundstripe.com/v1"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        license_profile: SoundtrackLicense | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self._api_key = api_key or get_setting("SOUNDSTRIPE_API_KEY")
        if not self._api_key:
            raise ValueError("Set SOUNDSTRIPE_API_KEY or pass api_key explicitly")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._license_profile = license_profile or SoundtrackLicense()
        self._timeout_seconds = timeout_seconds

    async def search(self, request: SoundtrackSearchRequest) -> SoundtrackSearchResponse:
        """Search the official songs endpoint and normalize its JSON:API response."""
        payload = await asyncio.to_thread(self._request_json, self._build_params(request))
        return self._normalize_response(payload)

    def _build_params(self, request: SoundtrackSearchRequest) -> dict[str, str]:
        params = {
            "filter[q]": " ".join([request.query, *request.search_terms]).strip(),
            "page[size]": str(request.limit),
            "filter[duration][min]": str(
                max(1, int(request.target_duration_seconds - request.duration_tolerance_seconds))
            ),
            "filter[duration][max]": str(
                int(request.target_duration_seconds + request.duration_tolerance_seconds)
            ),
        }
        if request.tempo_min_bpm is not None:
            params["filter[bpm][min]"] = str(request.tempo_min_bpm)
        if request.tempo_max_bpm is not None:
            params["filter[bpm][max]"] = str(request.tempo_max_bpm)
        if request.instrumental_preferred:
            params["filter[instrumental]"] = "true"
        for field, values in (("mood", request.moods), ("genre", request.genres)):
            normalized = [value.strip() for value in values if value.strip()]
            if normalized:
                params[f"filter[tags][{field}]"] = ",".join(normalized)
        if request.cursor:
            try:
                page = int(request.cursor)
            except ValueError as exc:
                raise ValueError("Soundstripe cursor must be a page number") from exc
            if page < 1:
                raise ValueError("Soundstripe page number must be positive")
            params["page[number]"] = str(page)
        return params

    def _request_json(self, params: dict[str, str]) -> dict[str, Any]:
        url = f"{self.api_base_url}/songs?{urlencode(params)}"
        request = Request(
            url,
            headers={
                "Authorization": f"Token {self._api_key}",
                "Accept": "application/vnd.api+json",
            },
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise SoundstripeAPIError(
                f"Soundstripe API returned HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise SoundstripeAPIError(f"Soundstripe API request failed: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SoundstripeAPIError("Soundstripe API returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise SoundstripeAPIError("Soundstripe API response must be a JSON object")
        return payload

    def _normalize_response(self, payload: dict[str, Any]) -> SoundtrackSearchResponse:
        data, included = payload.get("data", []), payload.get("included", [])
        if not isinstance(data, list) or not isinstance(included, list):
            raise SoundstripeAPIError("Soundstripe response has invalid data/included fields")
        resources = {
            (item.get("type"), str(item.get("id"))): item
            for item in included
            if isinstance(item, dict)
        }
        candidates = []
        retrieved_at = datetime.now(UTC)
        for song in data:
            if not isinstance(song, dict):
                continue
            attributes = song.get("attributes") or {}
            relationships = song.get("relationships") or {}
            audio_refs = (relationships.get("audio_files") or {}).get("data") or []
            artist_refs = (relationships.get("artists") or {}).get("data") or []
            audio = self._related_resource(resources, audio_refs, "audio_files")
            artist = self._related_resource(resources, artist_refs, "artists")
            audio_attributes = (audio or {}).get("attributes") or {}
            artist_attributes = (artist or {}).get("attributes") or {}
            song_id = str(song.get("id", ""))
            duration = audio_attributes.get("duration")
            if not song_id or duration is None:
                continue
            tag_data = attributes.get("tags") or {}
            tags = sorted(
                {
                    str(tag)
                    for values in tag_data.values()
                    if isinstance(values, list)
                    for tag in values
                }
            )
            candidate = SoundtrackCandidate(
                candidate_id=f"soundstripe:{song_id}",
                origin="catalog",
                provider_id=self.provider_id,
                provider_asset_id=song_id,
                title=str(attributes.get("title") or f"Soundstripe track {song_id}"),
                creator=artist_attributes.get("name"),
                duration_seconds=float(duration),
                tempo_bpm=attributes.get("bpm"),
                instrumental=audio_attributes.get("instrumental"),
                tags=tags[:40],
                source_url=f"{self.api_base_url}/songs/{song_id}",
                preview_url=(audio_attributes.get("versions") or {}).get("mp3"),
                retrieved_at=retrieved_at,
                license=self._license_profile.model_copy(deep=True),
            )
            candidates.append(candidate)
        next_cursor = None
        next_url = (payload.get("links") or {}).get("next")
        if isinstance(next_url, str):
            pages = parse_qs(urlparse(next_url).query).get("page[number]", [])
            if pages:
                next_cursor = pages[0]
        return SoundtrackSearchResponse(
            provider_id=self.provider_id, candidates=candidates, next_cursor=next_cursor
        )

    @staticmethod
    def _related_resource(resources: dict, references: list, resource_type: str) -> dict | None:
        if not references:
            return None
        return resources.get((resource_type, str(references[0].get("id"))))
