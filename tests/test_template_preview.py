from types import SimpleNamespace

import pytest

from video_edit_agent.editing.planning_schemas import StoryTreatment
from video_edit_agent.rendering.template_preview import (
    TemplatePreviewError,
    _opening_overlays,
    _probe_dimensions,
    render_template_previews,
)
from video_edit_agent.templates import load_template


def test_opening_overlays_use_template_slots_and_separate_logline():
    template, _ = load_template("cinematic")
    treatment = StoryTreatment(
        working_title="A Journey North",
        logline="A quiet trip through the mountains.",
        narrative_arc=["Departure", "Return"],
        emotional_journey="Wonder",
    )

    overlays = _opening_overlays(template, treatment)

    assert [overlay.text for overlay in overlays] == [
        "A Journey North",
        "A quiet trip through the mountains.",
    ]
    assert overlays[0].decision == "approved"
    assert overlays[0].placement != overlays[1].placement
    assert all(overlay.animation == "none" for overlay in overlays)


def test_opening_overlays_reject_template_without_opening_slots():
    template = SimpleNamespace(id="empty", sequences={})
    treatment = SimpleNamespace(working_title="Title", logline="Logline")

    with pytest.raises(ValueError, match="no usable opening title slots"):
        _opening_overlays(template, treatment)


def test_probe_dimensions_reads_ffprobe_output(monkeypatch, tmp_path):
    import video_edit_agent.rendering.template_preview as preview

    monkeypatch.setattr(
        preview,
        "run_ffmpeg",
        lambda *args, **kwargs: SimpleNamespace(stdout='{"streams":[{"width":1280,"height":720}]}'),
    )

    assert _probe_dimensions("ffprobe", tmp_path / "sample.jpg") == (1280, 720)


def test_probe_dimensions_rejects_missing_video_stream(monkeypatch, tmp_path):
    import video_edit_agent.rendering.template_preview as preview

    monkeypatch.setattr(
        preview,
        "run_ffmpeg",
        lambda *args, **kwargs: SimpleNamespace(stdout='{"streams":[]}'),
    )

    with pytest.raises(TemplatePreviewError, match="could not determine"):
        _probe_dimensions("ffprobe", tmp_path / "sample.jpg")


def test_template_preview_requires_unique_two_or_three_ids(tmp_path):
    with pytest.raises(ValueError, match="exactly two or three"):
        render_template_previews(tmp_path, ["cinematic"])

    with pytest.raises(ValueError, match="unique"):
        render_template_previews(tmp_path, ["cinematic", "cinematic"])


def test_template_preview_output_must_stay_inside_project(tmp_path):
    outside = tmp_path.parent / "outside-previews"

    with pytest.raises(ValueError, match="inside the project folder"):
        render_template_previews(tmp_path, ["cinematic", "vlog"], outside)
