from pathlib import Path

from video_edit_agent.highlights.pipeline import _score, find_highlights_folder
from video_edit_agent.highlights.schemas import HighlightSignals


def test_visual_score_is_explainable_and_keeps_low_motion_scenery() -> None:
    signals = HighlightSignals(motion=0.0, sharpness=0.9, exposure=1.0, saturation=0.4)
    score, reasons = _score(signals)
    assert 0 <= score <= 100
    assert any("Low motion" in reason for reason in reasons)
    assert len(reasons) == 5


def test_highlight_pipeline_ranks_candidates_without_understanding(
    monkeypatch, tmp_path: Path
) -> None:
    from video_edit_agent.highlights import pipeline

    video = tmp_path / "clip.mp4"
    video.touch()
    info = type("Info", (), {"path": "clip.mp4", "duration_seconds": 12.0})()
    analysis = type("Analysis", (), {"video_count": 1, "videos": [info]})()
    monkeypatch.setattr(pipeline, "analyze_folder", lambda root: analysis)
    monkeypatch.setattr(pipeline, "discover_videos", lambda root: [video])
    monkeypatch.setattr(pipeline, "detect_scenes", lambda *args: [(0.0, 5.0), (5.0, 12.0)])
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"sample")
    monkeypatch.setattr(
        pipeline,
        "extract_representative_frames",
        lambda *args, **kwargs: [(1.0, frame), (2.0, frame)],
    )
    scores = iter(
        [
            HighlightSignals(motion=0.1, sharpness=0.3, exposure=0.5, saturation=0.2),
            HighlightSignals(motion=0.2, sharpness=0.9, exposure=0.9, saturation=0.8),
        ]
    )
    monkeypatch.setattr(pipeline, "_measure_frames", lambda paths: next(scores))

    report = find_highlights_folder(tmp_path)
    assert report.video_count == 1
    assert report.candidate_count == 2
    assert report.candidates[0].start_seconds == 5.0
    assert report.candidates[0].score > report.candidates[1].score
    assert all(candidate.review_required for candidate in report.candidates)
    assert all(candidate.video_path == "clip.mp4" for candidate in report.candidates)


def test_highlight_pipeline_rejects_invalid_sample_count(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="between 1 and 10"):
        find_highlights_folder(tmp_path, samples_per_scene=0)
