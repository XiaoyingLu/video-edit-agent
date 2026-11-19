import asyncio
import hashlib
import json
from urllib.error import HTTPError

import pytest

from travel_video_agent.soundtrack.providers import stable_audio
from travel_video_agent.soundtrack.providers.stable_audio import (
    StableAudioAPIError,
    StableAudioGenerationProvider,
    StableAudioGenerationRequest,
)


class FakeResponse:
    def __init__(self, status, payload):
        self.status = status
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


def test_generation_submits_multipart_and_polls_until_audio_is_ready(tmp_path, monkeypatch):
    generation_id = "a" * 64
    responses = iter(
        [
            FakeResponse(202, json.dumps({"id": generation_id}).encode()),
            FakeResponse(202, b'{"id":"still-processing","status":"in-progress"}'),
            FakeResponse(200, b"generated-mp3-bytes"),
        ]
    )
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        return next(responses)

    monkeypatch.setattr(stable_audio, "urlopen", fake_urlopen)
    monkeypatch.setattr(stable_audio.time, "sleep", lambda _: None)
    provider = StableAudioGenerationProvider(api_key="test-api-key")
    request = StableAudioGenerationRequest(
        prompt="A calm instrumental piano and strings travel score.",
        duration_seconds=60,
        seed=42,
        poll_interval_seconds=1,
    )

    result = asyncio.run(provider.generate(request, output_path=tmp_path / "track.mp3"))

    assert len(requests) == 3
    assert requests[0].method == "POST"
    assert requests[0].full_url.endswith("/audio/stable-audio/text-to-audio")
    assert requests[0].get_header("Authorization") == "Bearer test-api-key"
    assert b'name="model"\r\n\r\nstable-audio-3' in requests[0].data
    assert b'name="duration"\r\n\r\n60' in requests[0].data
    assert b'name="seed"\r\n\r\n42' in requests[0].data
    assert requests[1].full_url.endswith(f"/audio/results/{generation_id}")
    assert result.generation_id == generation_id
    assert result.output_path.read_bytes() == b"generated-mp3-bytes"
    assert result.sha256 == hashlib.sha256(b"generated-mp3-bytes").hexdigest()


def test_api_key_is_required(monkeypatch):
    monkeypatch.delenv("STABILITY_API_KEY", raising=False)
    with pytest.raises(ValueError, match="STABILITY_API_KEY"):
        StableAudioGenerationProvider(api_key="")


@pytest.mark.parametrize("duration", [0, 381])
def test_generation_duration_must_fit_provider_limit(duration):
    with pytest.raises(ValueError):
        StableAudioGenerationRequest(prompt="Ambient score", duration_seconds=duration)


def test_invalid_generation_id_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stable_audio,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(202, b'{"id":"not-a-generation-id"}'),
    )
    provider = StableAudioGenerationProvider(api_key="test-api-key")
    request = StableAudioGenerationRequest(prompt="Ambient score", duration_seconds=30)

    with pytest.raises(StableAudioAPIError, match="invalid generation ID"):
        provider._generate_sync(request, tmp_path / "track.mp3")


def test_http_error_does_not_expose_api_key(tmp_path, monkeypatch):
    error = HTTPError(
        "https://api.stability.ai/generate",
        401,
        "Unauthorized",
        {},
        None,
    )
    error.read = lambda: b'{"name":"unauthorized","errors":["Invalid API key"]}'
    monkeypatch.setattr(stable_audio, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(error))
    provider = StableAudioGenerationProvider(api_key="secret-test-key")
    request = StableAudioGenerationRequest(prompt="Ambient score", duration_seconds=30)

    with pytest.raises(StableAudioAPIError, match="HTTP 401") as exc_info:
        provider._generate_sync(request, tmp_path / "track.mp3")
    assert "secret-test-key" not in str(exc_info.value)


def test_existing_output_is_not_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "track.mp3"
    output.write_bytes(b"keep me")
    provider = StableAudioGenerationProvider(api_key="test-api-key")
    request = StableAudioGenerationRequest(prompt="Ambient score", duration_seconds=30)

    with pytest.raises(FileExistsError):
        provider._generate_sync(request, output)

    assert output.read_bytes() == b"keep me"
