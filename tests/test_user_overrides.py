import json

import pytest

from video_edit_agent.editing.timeline_schemas import DraftTimelineClip, ProjectDraftTimeline
from video_edit_agent.editing.user_overrides import (
    UserOverrideError,
    apply_user_overrides,
    load_user_overrides,
    save_user_overrides,
)


def _timeline(root):
    return ProjectDraftTimeline(
        project_path=str(root.resolve()),
        style="cinematic",
        pacing="moderate",
        target_duration_seconds=10,
        proposed_duration_seconds=8,
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


def test_overrides_are_stored_separately_and_reapplied_by_clip_id(tmp_path):
    save_user_overrides(
        tmp_path,
        {
            "mountain": {
                "source_start_seconds": 10.5,
                "source_end_seconds": 13.5,
                "sequence_index": 1,
                "caption": "Above the clouds",
                "gain_db": -3.0,
            },
            "coast": {
                "sequence_index": 2,
                "caption": "Coastal opening",
            },
        },
    )

    generated = _timeline(tmp_path)
    effective = apply_user_overrides(generated)

    assert [clip.clip_id for clip in effective.clips] == ["mountain", "coast"]
    assert effective.clips[0].source_start_seconds == 10.5
    assert effective.clips[0].duration_seconds == 3
    assert effective.clips[0].caption == "Above the clouds"
    assert effective.clips[0].gain_db == -3.0
    assert effective.clips[1].caption == "Coastal opening"
    assert [clip.clip_id for clip in generated.clips] == ["coast", "mountain"]
    assert generated.clips[0].source_start_seconds == 2

    manifest = tmp_path / ".video-agent" / "user-overrides.json"
    assert manifest.is_file()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["clips"]["mountain"]["gain_db"] == -3.0


def test_override_manifest_merges_fields_without_losing_existing_edits(tmp_path):
    save_user_overrides(tmp_path, {"coast": {"caption": "First caption"}})
    save_user_overrides(tmp_path, {"coast": {"gain_db": 2.5}})

    assert load_user_overrides(tmp_path)["coast"] == {
        "caption": "First caption",
        "gain_db": 2.5,
    }


def test_reapplied_trim_must_fit_regenerated_source_interval(tmp_path):
    save_user_overrides(
        tmp_path,
        {"coast": {"source_start_seconds": 1.5, "source_end_seconds": 5.5}},
    )
    with pytest.raises(UserOverrideError, match="no longer fits"):
        apply_user_overrides(_timeline(tmp_path))


def test_unknown_clip_id_is_not_silently_discarded(tmp_path):
    save_user_overrides(tmp_path, {"removed-shot": {"caption": "Keep this"}})
    with pytest.raises(UserOverrideError, match="no longer present"):
        apply_user_overrides(_timeline(tmp_path))


def test_invalid_override_fields_are_rejected(tmp_path):
    with pytest.raises(UserOverrideError, match="Unsupported override"):
        save_user_overrides(tmp_path, {"coast": {"arbitrary": "value"}})


def test_override_values_are_validated_against_clip_schema(tmp_path):
    save_user_overrides(tmp_path, {"coast": {"gain_db": 30.0}})
    with pytest.raises(UserOverrideError, match="Invalid saved override"):
        apply_user_overrides(_timeline(tmp_path))


def test_reapplying_already_materialized_overrides_preserves_approval(tmp_path):
    timeline = _timeline(tmp_path)
    timeline.clips[0].source_start_seconds = 2.5
    timeline.clips[0].duration_seconds = 3.5
    timeline.proposed_duration_seconds = 7.5
    timeline.clips[0].decision = "approved"
    timeline.clips[1].decision = "rejected"
    save_user_overrides(
        tmp_path,
        {"coast": {"source_start_seconds": 2.5, "source_end_seconds": 6.0}},
    )

    effective = apply_user_overrides(timeline)

    assert effective.timeline_approved is timeline.timeline_approved
    assert effective.approved_duration_seconds == timeline.approved_duration_seconds
    assert effective.clips[0].decision == "approved"
