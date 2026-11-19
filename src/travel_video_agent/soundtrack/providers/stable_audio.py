"""Stable Audio 3.0 asynchronous text-to-audio API adapter."""

import asyncio
import hashlib
import json
import os
import re
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


class StableAudioAPIError(RuntimeError):
    """Raised when Stability AI rejects a request or returns an invalid response."""


class StableAudioGenerationRequest(BaseModel):
    """Validated prompt and generation settings for Stable Audio 3.0."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=10_000)
    duration_seconds: float = Field(ge=1, le=380)
    output_format: Literal["mp3", "wav"] = "mp3"
    seed: int | None = Field(default=None, ge=0, le=4_294_967_294)
    steps: int = Field(default=8, ge=4, le=8)
    cfg_scale: float = Field(default=1, ge=1, le=25)
    poll_interval_seconds: float = Field(default=10, gt=0, le=60)
    timeout_seconds: float = Field(default=900, gt=0, le=3600)


class StableAudioGenerationResult(BaseModel):
    """Completed audio file and provider provenance."""

    model_config = ConfigDict(extra="forbid")

    provider_id: Literal["stable_audio"] = "stable_audio"
    generation_id: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    output_path: Path
    output_format: Literal["mp3", "wav"]
    requested_duration_seconds: float = Field(gt=0)
    seed: int | None = None
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    generated_at: datetime
    prompt: str
    source_url: str = "https://platform.stability.ai/"
    license_name: str = "Stability AI Stable Audio API commercial-use terms"
    license_url: str = (
        "https://kb.stability.ai/knowledge-base/stable-audio-commerical-and-usage-licensing"
    )
    evidence_url: str = "https://docs.stability.ai/api-reference/stable-audio/text-to-audio"


class StableAudioGenerationProvider:
    """Generate audio through the official Stable Audio 3.0 async REST API."""

    provider_id = "stable_audio"
    api_base_url = "https://api.stability.ai/v2beta"
    generation_endpoint = "/audio/stable-audio/text-to-audio"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        request_timeout_seconds: float = 60,
    ) -> None:
        self._api_key = api_key or os.getenv("STABILITY_API_KEY")
        if not self._api_key:
            raise ValueError("Set STABILITY_API_KEY or pass api_key explicitly")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        self._request_timeout_seconds = request_timeout_seconds

    async def generate(
        self,
        request: StableAudioGenerationRequest,
        *,
        output_path: Path,
    ) -> StableAudioGenerationResult:
        """Run the blocking submit/poll/download lifecycle off the event loop."""
        return await asyncio.to_thread(self._generate_sync, request, output_path)

    def _generate_sync(
        self,
        request: StableAudioGenerationRequest,
        output_path: Path,
    ) -> StableAudioGenerationResult:
        destination = output_path.expanduser().resolve()
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite generated audio: {destination}")
        generation_id = self._start_generation(request)
        audio_bytes = self._wait_for_result(generation_id, request)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{generation_id}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(audio_bytes)
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        digest = hashlib.sha256(audio_bytes).hexdigest()
        return StableAudioGenerationResult(
            generation_id=generation_id,
            output_path=destination,
            output_format=request.output_format,
            requested_duration_seconds=request.duration_seconds,
            seed=request.seed,
            sha256=digest,
            generated_at=datetime.now(UTC),
            prompt=request.prompt,
        )

    def _start_generation(self, request: StableAudioGenerationRequest) -> str:
        boundary = f"video-agent-{secrets.token_hex(16)}"
        fields = {
            "model": "stable-audio-3",
            "prompt": request.prompt,
            "duration": str(request.duration_seconds),
            "output_format": request.output_format,
            "steps": str(request.steps),
            "cfg_scale": str(request.cfg_scale),
        }
        if request.seed is not None:
            fields["seed"] = str(request.seed)
        body = self._encode_multipart(fields, boundary)
        http_request = Request(
            f"{self.api_base_url}{self.generation_endpoint}",
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=self._request_timeout_seconds) as response:
                status = response.status
                payload = response.read()
        except HTTPError as exc:
            self._raise_http_error(exc)
        except (URLError, TimeoutError) as exc:
            raise StableAudioAPIError(f"Stable Audio generation request failed: {exc}") from exc
        if status != 202:
            raise StableAudioAPIError(
                f"Stable Audio expected HTTP 202 when starting generation, received {status}"
            )
        try:
            generation_id = json.loads(payload.decode("utf-8"))["id"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise StableAudioAPIError("Stable Audio returned an invalid generation response") from exc
        if not isinstance(generation_id, str) or not re.fullmatch(
            r"[a-fA-F0-9]{64}", generation_id
        ):
            raise StableAudioAPIError("Stable Audio returned an invalid generation ID")
        return generation_id

    def _wait_for_result(
        self,
        generation_id: str,
        request: StableAudioGenerationRequest,
    ) -> bytes:
        deadline = time.monotonic() + request.timeout_seconds
        result_url = f"{self.api_base_url}/audio/results/{generation_id}"
        while True:
            http_request = Request(
                result_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Accept": "audio/*",
                },
                method="GET",
            )
            try:
                with urlopen(http_request, timeout=self._request_timeout_seconds) as response:
                    status = response.status
                    payload = response.read()
            except HTTPError as exc:
                self._raise_http_error(exc)
            except (URLError, TimeoutError) as exc:
                raise StableAudioAPIError(f"Stable Audio result polling failed: {exc}") from exc
            if status == 200:
                if not payload:
                    raise StableAudioAPIError("Stable Audio returned an empty audio file")
                return payload
            if status != 202:
                raise StableAudioAPIError(
                    f"Stable Audio result polling returned unexpected HTTP {status}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Stable Audio generation {generation_id} exceeded "
                    f"{request.timeout_seconds:g} seconds"
                )
            time.sleep(min(request.poll_interval_seconds, remaining))

    @staticmethod
    def _encode_multipart(fields: dict[str, str], boundary: str) -> bytes:
        chunks = []
        for name, value in fields.items():
            chunks.extend(
                [
                    f"--{boundary}\r\n".encode(),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                    value.encode("utf-8"),
                    b"\r\n",
                ]
            )
        chunks.append(f"--{boundary}--\r\n".encode())
        return b"".join(chunks)

    @staticmethod
    def _raise_http_error(exc: HTTPError) -> None:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise StableAudioAPIError(
            f"Stable Audio API returned HTTP {exc.code}: {detail}"
        ) from exc
