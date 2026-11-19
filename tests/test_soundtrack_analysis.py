import hashlib
import json
from datetime import UTC, datetime

import pytest

from travel_video_agent.artifact_tracking import register_artifact
from travel_video_agent.music.schemas import MusicAnalysis
from travel_video_agent.soundtrack import analysis as soundtrack_analysis
from travel_video_agent.soundtrack.analysis import (
    SelectedSoundtrackAnalysisError,
    analyze_selected_soundtrack,
)
from travel_video_agent.soundtrack.catalog import SoundtrackSearchResponse
from travel_video_agent.soundtrack.review import (
    build_soundtrack_candidate_review,
    save_soundtrack_candidate_review,
    select_soundtrack_candidate,
)
from travel_video_agent.soundtrack.schemas import (
    SoundtrackBrief,
    SoundtrackCandidate,
    SoundtrackLicense,
)


def _candidate(*, origin="catalog", asset_path=None, checksum=None):
    return SoundtrackCandidate(
        candidate_id="track-1",
        origin=origin,
        provider_id="test-provider",
        provider_asset_id="provider-track-1",
        title="Mountain Light",
        creator="Example Artist",
        duration_seconds=60,
        tempo_bpm=90,
        instrumental=True,
        tags=["calm", "ambient", "piano"],
        source_url="https://example.com/track",
        preview_url="https://example.com/preview.mp3",
        asset_path=asset_path,
        retrieved_at=datetime.now(UTC),
        asset_sha256=checksum,
        license=SoundtrackLicense(
            license_type="custom",
            license_name="Verified commercial agreement",
            license_url="https://example.com/license",
            evidence_url="https://example.com/terms",
            commercial_use_allowed=True,
            monetized_video_allowed=True,
            derivatives_allowed=True,
            synchronization_allowed=True,
        ),
    )


def _select(root, candidate):
    artifact_dir = root / ".video-agent"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    brief = SoundtrackBrief(
        project_path=str(root.resolve()),
        source_video_plan=".video-agent/video-project-plan.json",
        target_duration_seconds=60,
        video_style="cinematic",
        narrative_summary="A quiet mountain journey ending at a summit.",
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
    brief_path = artifact_dir / "soundtrack-brief.json"
    brief_path.write_text(brief.model_dump_json(), encoding="utf-8")
    candidates_path = artifact_dir / "soundtrack-candidates.json"
    candidates_path.write_text(
        SoundtrackSearchResponse(
            provider_id="test-provider", candidates=[candidate]
        ).model_dump_json(),
        encoding="utf-8",
    )
    register_artifact(root, candidates_path, dependencies=[brief_path])
    report = build_soundtrack_candidate_review(root, candidates_path=candidates_path)
    save_soundtrack_candidate_review(root, report)
    selection, selection_path = select_soundtrack_candidate(root, candidate_id="track-1")
    return selection, selection_path


def _analysis(root, music_path):
    return MusicAnalysis(
        project_path=str(root.resolve()),
        music_path=str(music_path.resolve()),
        duration_seconds=60,
        sample_rate=11025,
        bpm=90,
        beat_confidence=0.8,
        beat_interval_seconds=0.6667,
        beat_times_seconds=[0.1, 0.767, 1.433],
        downbeat_times_seconds=[0.1],
        sections=[],
        suggested_cut_points_seconds=[0.1, 0.767, 1.433],
    )


def test_analyze_selected_catalog_track_records_provenance_and_dependencies(tmp_path, monkeypatch):
    _, selection_path = _select(tmp_path, _candidate())
    audio = tmp_path / ".video-agent" / "licensed-catalog-track.mp3"
    audio.write_bytes(b"locally imported licensed audio")
    monkeypatch.setattr(
        soundtrack_analysis, "analyze_music", lambda root, path: _analysis(root, path)
    )

    report, record, analysis_path, record_path = analyze_selected_soundtrack(
        tmp_path, music_path=audio.relative_to(tmp_path)
    )

    assert report.bpm == 90
    assert record.candidate_id == "track-1"
    assert record.origin == "catalog"
    assert record.audio_sha256 == hashlib.sha256(audio.read_bytes()).hexdigest()
    assert json.loads(analysis_path.read_text())["music_path"] == str(audio.resolve())
    assert json.loads(record_path.read_text())["selection_path"] == selection_path.relative_to(
        tmp_path
    ).as_posix()
    manifest = json.loads((tmp_path / ".video-agent" / "artifact-manifest.json").read_text())
    analysis_dependencies = manifest["artifacts"][analysis_path.relative_to(tmp_path).as_posix()][
        "dependencies"
    ]
    assert selection_path.relative_to(tmp_path).as_posix() in analysis_dependencies
    assert audio.relative_to(tmp_path).as_posix() in analysis_dependencies


def test_catalog_selection_requires_local_audio_not_preview_url(tmp_path):
    _select(tmp_path, _candidate())

    with pytest.raises(SelectedSoundtrackAnalysisError, match="require --music"):
        analyze_selected_soundtrack(tmp_path)


def test_generated_selection_uses_recorded_asset_and_checks_checksum(tmp_path, monkeypatch):
    audio = tmp_path / ".video-agent" / "generated-soundtracks" / "generated.mp3"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"generated track")
    checksum = hashlib.sha256(audio.read_bytes()).hexdigest()
    candidate = _candidate(
        origin="generated",
        asset_path=audio.relative_to(tmp_path).as_posix(),
        checksum=checksum,
    )
    _select(tmp_path, candidate)
    monkeypatch.setattr(
        soundtrack_analysis, "analyze_music", lambda root, path: _analysis(root, path)
    )

    _, record, _, _ = analyze_selected_soundtrack(tmp_path)

    assert record.music_path == audio.relative_to(tmp_path).as_posix()


def test_generated_selection_rejects_tampered_audio(tmp_path):
    audio = tmp_path / ".video-agent" / "generated-soundtracks" / "generated.mp3"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"original")
    candidate = _candidate(
        origin="generated",
        asset_path=audio.relative_to(tmp_path).as_posix(),
        checksum=hashlib.sha256(audio.read_bytes()).hexdigest(),
    )
    _select(tmp_path, candidate)
    audio.write_bytes(b"tampered")

    with pytest.raises(ValueError):
        analyze_selected_soundtrack(tmp_path)


def test_selected_audio_must_be_inside_project(tmp_path):
    _select(tmp_path, _candidate())

    with pytest.raises(SelectedSoundtrackAnalysisError, match="inside the project"):
        analyze_selected_soundtrack(tmp_path, music_path=tmp_path.parent / "outside.mp3")
