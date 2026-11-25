import json
from pathlib import Path

import pytest

from video_edit_agent.editing.footage_visual_review import (
    generate_footage_visual_review,
    load_footage_review_selections,
)
from test_footage_review import _artifacts


def test_generate_footage_visual_review_writes_candidate_choices(tmp_path: Path) -> None:
    plan_path, suggestions_path = _artifacts(tmp_path)

    destination, shot_count, candidate_count = generate_footage_visual_review(
        tmp_path, plan_path=plan_path, suggestions_path=suggestions_path
    )

    document = destination.read_text(encoding="utf-8")
    assert destination == tmp_path / ".video-agent" / "footage-review.html"
    assert shot_count == 1
    assert candidate_count == 1
    assert "Option 1" in document
    assert "Leave this shot unassigned" in document
    assert "Download selections" in document
    assert "nothing is approved or rendered" in document.lower()


def test_load_footage_review_selections_requires_every_shot(tmp_path: Path) -> None:
    path = tmp_path / "selections.json"
    path.write_text(
        json.dumps({"schema_version": "1.0", "selections": {"shot-001": 2, "shot-002": 0}}),
        encoding="utf-8",
    )

    selected, rejected = load_footage_review_selections(
        path, expected_shot_ids={"shot-001", "shot-002"}
    )

    assert selected == {"shot-001": 2}
    assert rejected == {"shot-002"}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"schema_version": "2.0", "selections": {"shot-001": 1}},
        {"schema_version": "1.0", "selections": {"shot-001": "1"}},
        {"schema_version": "1.0", "selections": {"shot-001": 11}},
    ],
)
def test_load_footage_review_selections_rejects_invalid_exports(tmp_path: Path, payload) -> None:
    path = tmp_path / "selections.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_footage_review_selections(path, expected_shot_ids={"shot-001"})
