import hashlib
import json
from datetime import UTC, datetime

import pytest

from travel_video_agent.music.schemas import MusicAnalysis
from travel_video_agent.soundtrack.generated import (
    GeneratedCandidateError,
    prepare_generated_soundtrack_candidates,
)
from travel_video_agent.soundtrack.generation import GeneratedSoundtrackArtifact
from travel_video_agent.soundtrack.providers.stable_audio import StableAudioGenerationResult
from travel_video_agent.soundtrack.review import (
    build_soundtrack_candidate_review,
    save_soundtrack_candidate_review,
)
from travel_video_agent.soundtrack.schemas import SoundtrackBrief
from travel_video_agent.soundtrack import generated


def _write_brief(root):
    artifact_dir = root / ".video-agent"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    brief = SoundtrackBrief(
        project_path=str(root.resolve()),
        source_video_plan=".video-agent/video-project-plan.json",
        target_duration_seconds=12,
        video_style="cinematic",
        narrative_summary="A quiet mountain journey ending at a summit.",
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
                "end_seconds": 12,
                "energy": "medium",
            }
        ],
        search_terms=["calm", "ambient"],
        generation_prompt="A calm cinematic instrumental.",
    )
    path = artifact_dir / "soundtrack-brief.json"
    path.write_text(brief.model_dump_json(), encoding="utf-8")
    return path


def _write_generated_track(root):
    generated_dir = root / ".video-agent" / "generated-soundtracks"
    generated_dir.mkdir(parents=True, exist_ok=True)
    audio = generated_dir / "track-12345678.mp3"
    payload = b"generated test audio"
    audio.write_bytes(payload)
    result = StableAudioGenerationResult(
        generation_id="a" * 64,
        output_path=audio,
        output_format="mp3",
        requested_duration_seconds=12,
        seed=7,
        sha256=hashlib.sha256(payload).hexdigest(),
        generated_at=datetime.now(UTC),
        prompt="A calm cinematic instrumental.",
    )
    artifact = GeneratedSoundtrackArtifact(
        project_path=str(root.resolve()),
        brief_path=".video-agent/soundtrack-brief.json",
        audio_path=audio.relative_to(root).as_posix(),
        provider_result=result,
        license={
            "license_type": "custom",
            "license_name": "Stability AI Stable Audio API commercial-use terms",
            "license_url": "https://kb.stability.ai/knowledge-base/stable-audio-commerical-and-usage-licensing",
            "evidence_url": "https://docs.stability.ai/api-reference/stable-audio/text-to-audio",
            "commercial_use_allowed": True,
            "monetized_video_allowed": True,
            "derivatives_allowed": True,
            "synchronization_allowed": True,
        },
        generated_at=datetime.now(UTC),
    )
    metadata = audio.with_suffix(".json")
    metadata.write_text(artifact.model_dump_json(), encoding="utf-8")
    return audio, metadata


def _analysis(audio_path):
    return MusicAnalysis(
        project_path=str(audio_path.parent.parent.parent.resolve()),
        music_path=str(audio_path.resolve()),
        duration_seconds=12.0,
        sample_rate=11025,
        bpm=92.4,
        beat_confidence=0.82,
        beat_interval_seconds=0.6494,
        beat_times_seconds=[0.1, 0.75, 1.4],
        downbeat_times_seconds=[0.1],
        sections=[
            {"start_seconds": 0, "end_seconds": 12, "energy": "medium", "energy_rms": 0.2}
        ],
        suggested_cut_points_seconds=[0.1, 0.75, 1.4],
    )


def test_prepare_candidates_analyzes_generated_audio_and_records_provenance(tmp_path, monkeypatch):
    _write_brief(tmp_path)
    audio, metadata = _write_generated_track(tmp_path)
    monkeypatch.setattr(generated, "analyze_music", lambda root, path: _analysis(path))

    collection, destination = prepare_generated_soundtrack_candidates(tmp_path)

    candidate = collection.candidates[0]
    assert candidate.origin == "generated"
    assert candidate.provider_id == "stability_stable_audio"
    assert candidate.asset_path == audio.relative_to(tmp_path).as_posix()
    assert candidate.tempo_bpm == 92
    assert candidate.duration_seconds == 12
    assert candidate.asset_sha256 == hashlib.sha256(audio.read_bytes()).hexdigest()
    manifest = json.loads(
        (tmp_path / ".video-agent" / "artifact-manifest.json").read_text()
    )
    dependencies = manifest["artifacts"][destination.relative_to(tmp_path).as_posix()][
        "dependencies"
    ]
    assert metadata.relative_to(tmp_path).as_posix() in dependencies
    assert audio.relative_to(tmp_path).as_posix() in dependencies
    assert json.loads(destination.read_text())["candidates"][0]["origin"] == "generated"


def test_generated_candidate_is_playable_in_review_html(tmp_path, monkeypatch):
    _write_brief(tmp_path)
    audio, _ = _write_generated_track(tmp_path)
    monkeypatch.setattr(generated, "analyze_music", lambda root, path: _analysis(path))
    _, candidates_path = prepare_generated_soundtrack_candidates(tmp_path)

    report = build_soundtrack_candidate_review(tmp_path, candidates_path=candidates_path)
    _, html_path = save_soundtrack_candidate_review(tmp_path, report)
    page = html_path.read_text(encoding="utf-8")

    assert '<audio controls preload="none"' in page
    assert "generated-soundtracks/track-12345678.mp3" in page
    assert report.items[0].ranking.rights.status == "eligible"


def test_prepare_candidates_rejects_audio_checksum_mismatch(tmp_path, monkeypatch):
    _write_brief(tmp_path)
    audio, _ = _write_generated_track(tmp_path)
    audio.write_bytes(b"tampered")
    monkeypatch.setattr(generated, "analyze_music", lambda root, path: _analysis(path))

    with pytest.raises(GeneratedCandidateError, match="checksum mismatch"):
        prepare_generated_soundtrack_candidates(tmp_path)


def test_prepare_candidates_combines_catalog_and_generated_tracks(tmp_path, monkeypatch):
    from travel_video_agent.soundtrack.catalog import SoundtrackSearchResponse

    _write_brief(tmp_path)
    _write_generated_track(tmp_path)
    monkeypatch.setattr(generated, "analyze_music", lambda root, path: _analysis(path))
    catalog_path = tmp_path / ".video-agent" / "soundtrack-candidates.json"
    catalog_path.write_text(
        SoundtrackSearchResponse(provider_id="soundstripe", candidates=[]).model_dump_json(),
        encoding="utf-8",
    )

    collection, _ = prepare_generated_soundtrack_candidates(
        tmp_path, catalog_candidates_path=catalog_path
    )

    assert [candidate.origin for candidate in collection.candidates] == ["generated"]
    assert collection.catalog_candidates_path == catalog_path.relative_to(tmp_path).as_posix()
