import json
from pathlib import Path

import pytest

from video_edit_agent.editing.timeline_schemas import DraftTimelineClip, ProjectDraftTimeline
from video_edit_agent.rendering.social_crop_anchors import load_crop_anchors, save_crop_anchors


def _approved_timeline(root: Path) -> Path:
    directory = root / ".video-agent"
    directory.mkdir(parents=True)
    timeline = ProjectDraftTimeline(
        project_path=str(root.resolve()),
        style="cinematic",
        pacing="moderate",
        target_duration_seconds=4,
        proposed_duration_seconds=4,
        approved_duration_seconds=4,
        proposed_clip_count=1,
        timeline_approved=True,
        clips=[
            DraftTimelineClip(
                clip_id="coast-shot",
                sequence_index=1,
                video_path="coast.mp4",
                source_start_seconds=0,
                source_end_seconds=4,
                duration_seconds=4,
                combined_score=90,
                pacing_note="Opening",
                decision="approved",
            )
        ],
    )
    path = directory / "optimized-timeline.json"
    path.write_text(timeline.model_dump_json(), encoding="utf-8")
    return path


def test_save_crop_anchors_writes_loadable_timeline_bound_template(tmp_path):
    timeline_path = _approved_timeline(tmp_path)

    destination = save_crop_anchors(
        tmp_path,
        timeline_path=timeline_path,
        anchors=[{"clip_id": "coast-shot", "x": 0.72, "y": 0.38}],
    )

    document = json.loads(destination.read_text(encoding="utf-8"))
    timeline = ProjectDraftTimeline.model_validate_json(timeline_path.read_text())
    assert document["schema_version"] == "1.0"
    assert document["source_timeline"] == ".video-agent/optimized-timeline.json"
    assert document["anchors"] == [{"clip_id": "coast-shot", "x": 0.72, "y": 0.38}]
    assert load_crop_anchors(
        tmp_path, timeline_path, timeline, destination
    ) == {"coast-shot": (0.72, 0.38)}


def test_save_crop_anchors_rejects_invalid_coordinates_and_clip_ids(tmp_path):
    timeline_path = _approved_timeline(tmp_path)

    with pytest.raises(ValueError, match="between 0 and 1"):
        save_crop_anchors(
            tmp_path,
            timeline_path=timeline_path,
            anchors=[{"clip_id": "coast-shot", "x": 1.2, "y": 0.5}],
        )

    with pytest.raises(ValueError, match="unknown crop-anchor clip_id"):
        save_crop_anchors(
            tmp_path,
            timeline_path=timeline_path,
            anchors=[{"clip_id": "missing", "x": 0.5, "y": 0.5}],
        )


def test_save_crop_anchors_requires_approved_timeline(tmp_path):
    timeline_path = _approved_timeline(tmp_path)
    timeline = ProjectDraftTimeline.model_validate_json(timeline_path.read_text())
    timeline.timeline_approved = False
    timeline_path.write_text(timeline.model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="approve the complete timeline"):
        save_crop_anchors(
            tmp_path,
            timeline_path=timeline_path,
            anchors=[{"clip_id": "coast-shot", "x": 0.5, "y": 0.5}],
        )


def test_save_crop_anchors_rejects_output_outside_project(tmp_path):
    timeline_path = _approved_timeline(tmp_path)

    with pytest.raises(ValueError, match="inside the project folder"):
        save_crop_anchors(
            tmp_path,
            timeline_path=timeline_path,
            output_path=tmp_path.parent / "outside.json",
            anchors=[],
        )
