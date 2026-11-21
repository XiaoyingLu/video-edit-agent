import asyncio
import hashlib
import json
from datetime import UTC, datetime

import pytest

from video_edit_agent.soundtrack import generation
from video_edit_agent.soundtrack.generation import (
    SoundtrackGenerationError,
    generate_soundtrack_from_brief,
    stable_audio_license,
)
from video_edit_agent.soundtrack.providers.stable_audio import StableAudioGenerationResult
from video_edit_agent.soundtrack.schemas import SoundtrackBrief


def write_brief(root, *, duration=60):
    artifact_dir = root / ".video-agent"
    artifact_dir.mkdir(exist_ok=True)
    brief = SoundtrackBrief(
        project_path=str(root.resolve()),
        source_video_plan=".video-agent/video-project-plan.json",
        target_duration_seconds=duration,
        video_style="cinematic",
        narrative_summary="A quiet mountain journey ending with a summit reveal.",
        mood=["calm", "wonder"],
        genre_directions=["ambient", "cinematic"],
        instrumentation=["piano", "strings"],
        vocals="instrumental_preferred",
        tempo_min_bpm=70,
        tempo_max_bpm=110,
        energy_arc=[
            {
                "scene_id": "scene-1",
                "scene_title": "Journey",
                "start_seconds": 0,
                "end_seconds": duration,
                "energy": "medium",
            }
        ],
        search_terms=["calm", "ambient"],
        generation_prompt="A calm cinematic instrumental with piano and strings.",
    )
    path = artifact_dir / "soundtrack-brief.json"
    path.write_text(brief.model_dump_json(), encoding="utf-8")
    return path


def test_stable_audio_license_records_monetized_video_and_sync_rights():
    license_profile = stable_audio_license()

    assert license_profile.commercial_use_allowed is True
    assert license_profile.monetized_video_allowed is True
    assert license_profile.derivatives_allowed is True
    assert license_profile.synchronization_allowed is True
    assert "monetized YouTube" in license_profile.terms_snapshot
    assert license_profile.license_url.host == "kb.stability.ai"


def test_generation_persists_audio_metadata_and_artifact_dependencies(tmp_path, monkeypatch):
    brief_path = write_brief(tmp_path)
    audio_bytes = b"generated-audio-content"

    class FakeProvider:
        def __init__(self):
            pass

        async def generate(self, request, *, output_path):
            assert request.prompt.startswith("A calm cinematic")
            assert request.duration_seconds == 60
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(audio_bytes)
            return StableAudioGenerationResult(
                generation_id="b" * 64,
                output_path=output_path,
                output_format="mp3",
                requested_duration_seconds=60,
                seed=7,
                sha256=hashlib.sha256(audio_bytes).hexdigest(),
                generated_at=datetime.now(UTC),
                prompt=request.prompt,
            )

    monkeypatch.setattr(generation, "StableAudioGenerationProvider", FakeProvider)

    artifact, metadata_path = asyncio.run(generate_soundtrack_from_brief(tmp_path, seed=7))

    audio_path = tmp_path / artifact.audio_path
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        (tmp_path / ".video-agent" / "artifact-manifest.json").read_text(encoding="utf-8")
    )

    assert audio_path.read_bytes() == audio_bytes
    assert metadata["provider_result"]["generation_id"] == "b" * 64
    assert metadata["license"]["monetized_video_allowed"] is True
    assert metadata["human_review_required"] is True
    assert metadata["selected"] is False
    assert metadata["timeline_modified"] is False
    assert artifact.brief_path == brief_path.relative_to(tmp_path).as_posix()
    assert artifact.audio_path in manifest["artifacts"]
    assert metadata_path.relative_to(tmp_path).as_posix() in manifest["artifacts"]
    assert manifest["artifacts"][artifact.audio_path]["dependencies"] == [
        brief_path.relative_to(tmp_path).as_posix()
    ]


def test_generation_rejects_briefs_longer_than_provider_limit(tmp_path, monkeypatch):
    write_brief(tmp_path, duration=400)

    class UnexpectedProvider:
        def __init__(self):
            pytest.fail("provider must not be called for an unsupported duration")

    monkeypatch.setattr(generation, "StableAudioGenerationProvider", UnexpectedProvider)

    with pytest.raises(SoundtrackGenerationError, match="up to 380 seconds"):
        asyncio.run(generate_soundtrack_from_brief(tmp_path))


def test_generation_rejects_output_paths_outside_project(tmp_path, monkeypatch):
    write_brief(tmp_path)

    class UnexpectedProvider:
        def __init__(self):
            pytest.fail("provider must not be called for an unsafe output path")

    monkeypatch.setattr(generation, "StableAudioGenerationProvider", UnexpectedProvider)

    with pytest.raises(SoundtrackGenerationError, match="inside the project"):
        asyncio.run(
            generate_soundtrack_from_brief(tmp_path, output_path=tmp_path.parent / "outside.mp3")
        )


def test_generation_refuses_existing_metadata_before_provider_call(tmp_path, monkeypatch):
    write_brief(tmp_path)
    output = tmp_path / ".video-agent" / "generated-soundtracks" / "track.mp3"
    output.parent.mkdir(parents=True)
    output.with_suffix(".json").write_text("{}", encoding="utf-8")

    class UnexpectedProvider:
        def __init__(self):
            pytest.fail("provider must not be called when metadata already exists")

    monkeypatch.setattr(generation, "StableAudioGenerationProvider", UnexpectedProvider)

    with pytest.raises(FileExistsError):
        asyncio.run(generate_soundtrack_from_brief(tmp_path, output_path=output))
