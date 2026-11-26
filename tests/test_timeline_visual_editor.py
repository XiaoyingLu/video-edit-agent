import hashlib
import json
from pathlib import Path

import pytest

from video_edit_agent.editing.timeline_schemas import DraftTimelineClip, ProjectDraftTimeline
from video_edit_agent.editing.timeline_visual_editor import (
    TimelineVisualEditClip,
    TimelineVisualEditError,
    TimelineVisualEditExport,
    apply_timeline_visual_edits,
    load_timeline_visual_edits,
)


def _timeline(root: Path) -> Path:
    directory = root / ".video-agent"
    directory.mkdir(parents=True)
    timeline = ProjectDraftTimeline(
        project_path=str(root.resolve()),
        style="cinematic",
        pacing="moderate",
        target_duration_seconds=10,
        proposed_duration_seconds=8,
        approved_duration_seconds=0,
        proposed_clip_count=2,
        clips=[
            DraftTimelineClip(
                clip_id="coast",
                sequence_index=1,
                video_path="coast.mp4",
                source_start_seconds=2,
                source_end_seconds=6,
                duration_seconds=4,
                combined_score=90,
                pacing_note="Opening",
            ),
            DraftTimelineClip(
                clip_id="mountain",
                sequence_index=2,
                video_path="mountain.mp4",
                source_start_seconds=10,
                source_end_seconds=14,
                duration_seconds=4,
                combined_score=85,
                pacing_note="Follow-up",
            ),
        ],
    )
    path = directory / "optimized-timeline.json"
    path.write_text(timeline.model_dump_json(), encoding="utf-8")
    return path


def _export(path: Path, clips: list[TimelineVisualEditClip]) -> TimelineVisualEditExport:
    return TimelineVisualEditExport(
        timeline_path=".video-agent/optimized-timeline.json",
        timeline_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        clips=clips,
    )


def _patch_tracking(monkeypatch):
    monkeypatch.setattr(
        "video_edit_agent.editing.timeline_visual_editor.validate_artifact", lambda *_: True
    )
    monkeypatch.setattr(
        "video_edit_agent.editing.timeline_visual_editor.register_artifact",
        lambda *_args, **_kwargs: None,
    )


def test_apply_visual_edits_reorders_and_trims_into_separate_pending_revision(
    tmp_path, monkeypatch
):
    source = _timeline(tmp_path)
    _patch_tracking(monkeypatch)
    edits = _export(
        source,
        [
            TimelineVisualEditClip(
                clip_id="mountain",
                source_start_seconds=10.5,
                source_end_seconds=13.5,
                caption="Above the clouds",
                gain_db=-3.0,
            ),
            TimelineVisualEditClip(
                clip_id="coast", source_start_seconds=2.5, source_end_seconds=5.5
            ),
        ],
    )

    destination = apply_timeline_visual_edits(tmp_path, edits)

    revised = ProjectDraftTimeline.model_validate_json(destination.read_text())
    original = ProjectDraftTimeline.model_validate_json(source.read_text())
    assert [clip.clip_id for clip in revised.clips] == ["mountain", "coast"]
    assert [clip.sequence_index for clip in revised.clips] == [1, 2]
    assert revised.clips[0].source_start_seconds == 10.5
    assert revised.clips[0].duration_seconds == 3
    assert revised.clips[0].caption == "Above the clouds"
    assert revised.clips[0].gain_db == -3.0
    assert all(clip.decision == "pending" for clip in revised.clips)
    assert revised.timeline_approved is False
    assert revised.approved_duration_seconds == 0
    assert revised.proposed_duration_seconds == 6
    assert revised.clips[0].transition_in == "cut"
    assert [clip.clip_id for clip in original.clips] == ["coast", "mountain"]
    assert destination != source


def test_visual_edits_reject_stale_hash_and_incomplete_clip_set(tmp_path, monkeypatch):
    source = _timeline(tmp_path)
    _patch_tracking(monkeypatch)
    edits = _export(
        source,
        [TimelineVisualEditClip(clip_id="coast", source_start_seconds=2, source_end_seconds=6)],
    )
    with pytest.raises(TimelineVisualEditError, match="every timeline clip"):
        apply_timeline_visual_edits(tmp_path, edits)

    stale = edits.model_copy(update={"timeline_sha256": "0" * 64})
    with pytest.raises(TimelineVisualEditError, match="changed after visual editing"):
        apply_timeline_visual_edits(tmp_path, stale)


def test_visual_edits_reject_source_interval_extension_and_short_trim(tmp_path, monkeypatch):
    source = _timeline(tmp_path)
    _patch_tracking(monkeypatch)
    extended = _export(
        source,
        [
            TimelineVisualEditClip(clip_id="coast", source_start_seconds=1, source_end_seconds=5),
            TimelineVisualEditClip(
                clip_id="mountain", source_start_seconds=10, source_end_seconds=14
            ),
        ],
    )
    with pytest.raises(TimelineVisualEditError, match="within its original source interval"):
        apply_timeline_visual_edits(tmp_path, extended)

    too_short = _export(
        source,
        [
            TimelineVisualEditClip(clip_id="coast", source_start_seconds=2, source_end_seconds=2.5),
            TimelineVisualEditClip(
                clip_id="mountain", source_start_seconds=10, source_end_seconds=14
            ),
        ],
    )
    with pytest.raises(TimelineVisualEditError, match="minimum"):
        apply_timeline_visual_edits(tmp_path, too_short)


def test_load_visual_edits_rejects_duplicate_clip_ids(tmp_path):
    path = tmp_path / "edits.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "timeline_path": ".video-agent/optimized-timeline.json",
                "timeline_sha256": "a" * 64,
                "clips": [
                    {"clip_id": "coast", "source_start_seconds": 2, "source_end_seconds": 4},
                    {"clip_id": "coast", "source_start_seconds": 3, "source_end_seconds": 5},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TimelineVisualEditError, match="duplicate clip IDs"):
        load_timeline_visual_edits(path)


def test_visual_edits_preserve_transitions_when_order_is_unchanged(tmp_path, monkeypatch):
    source = _timeline(tmp_path)
    timeline = ProjectDraftTimeline.model_validate_json(source.read_text())
    timeline.clips[1].transition_in = "crossfade"
    timeline.clips[1].transition_duration_seconds = 0.5
    timeline.proposed_duration_seconds = 7.5
    source.write_text(timeline.model_dump_json(), encoding="utf-8")
    _patch_tracking(monkeypatch)
    edits = _export(
        source,
        [
            TimelineVisualEditClip(clip_id="coast", source_start_seconds=2, source_end_seconds=6),
            TimelineVisualEditClip(
                clip_id="mountain", source_start_seconds=10, source_end_seconds=14
            ),
        ],
    )

    destination = apply_timeline_visual_edits(tmp_path, edits)

    revised = ProjectDraftTimeline.model_validate_json(destination.read_text())
    assert revised.clips[1].transition_in == "crossfade"
    assert revised.clips[1].transition_duration_seconds == 0.5
    assert revised.proposed_duration_seconds == 7.5


def test_visual_edits_apply_selected_transition_without_reordering(tmp_path, monkeypatch):
    source = _timeline(tmp_path)
    _patch_tracking(monkeypatch)
    edits = _export(
        source,
        [
            TimelineVisualEditClip(
                clip_id="coast",
                source_start_seconds=2,
                source_end_seconds=6,
                transition_in="cut",
                transition_duration_seconds=0,
            ),
            TimelineVisualEditClip(
                clip_id="mountain",
                source_start_seconds=10,
                source_end_seconds=14,
                transition_in="dissolve",
                transition_duration_seconds=0.8,
            ),
        ],
    )

    destination = apply_timeline_visual_edits(tmp_path, edits)

    revised = ProjectDraftTimeline.model_validate_json(destination.read_text())
    assert revised.clips[0].transition_in == "cut"
    assert revised.clips[1].transition_in == "dissolve"
    assert revised.clips[1].transition_duration_seconds == 0.8
    assert revised.proposed_duration_seconds == 7.2


def test_visual_edits_force_first_clip_to_cut_and_limit_transition_duration(tmp_path, monkeypatch):
    source = _timeline(tmp_path)
    _patch_tracking(monkeypatch)
    edits = _export(
        source,
        [
            TimelineVisualEditClip(
                clip_id="coast",
                source_start_seconds=2,
                source_end_seconds=6,
                transition_in="fadeblack",
                transition_duration_seconds=0.5,
            ),
            TimelineVisualEditClip(
                clip_id="mountain",
                source_start_seconds=10,
                source_end_seconds=14,
                transition_in="crossfade",
                transition_duration_seconds=2,
            ),
        ],
    )

    destination = apply_timeline_visual_edits(tmp_path, edits)

    revised = ProjectDraftTimeline.model_validate_json(destination.read_text())
    assert revised.clips[0].transition_in == "cut"
    assert revised.clips[0].transition_duration_seconds == 0
    assert revised.clips[1].transition_duration_seconds == 1
