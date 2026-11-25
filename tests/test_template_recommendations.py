from pathlib import Path

import pytest

from video_edit_agent.template_recommendations import recommend_templates


def _write_template(project: Path, template_id: str, tags: str) -> None:
    directory = project / ".video-agent" / "templates"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{template_id}.yaml").write_text(
        f"id: {template_id}\nversion: 1\ntags:\n{tags}", encoding="utf-8"
    )


def test_recommendations_rank_use_case_matches_above_mood_matches(tmp_path: Path):
    _write_template(tmp_path, "mood-only", "  mood: [warm]\n")
    _write_template(tmp_path, "use-case", "  best_for: [travel]\n")
    _write_template(tmp_path, "both", "  mood: [warm]\n  best_for: [travel]\n")
    results = recommend_templates(moods=["WARM"], best_for=["travel"], project_root=tmp_path)
    assert [item.template.id for item in results] == ["both", "use-case", "mood-only"]
    assert results[0].score == 3
    assert results[0].matches == ("best_for:travel", "mood:warm")


def test_recommendations_normalize_hyphens_and_spaces(tmp_path: Path):
    _write_template(tmp_path, "daily", "  best_for: [daily-life]\n")
    results = recommend_templates(best_for=["Daily Life"], project_root=tmp_path)
    assert [item.template.id for item in results] == ["daily"]


def test_recommendations_return_at_most_three_and_ignore_unmatched_tags(tmp_path: Path):
    for template_id in ("one", "two", "three", "four"):
        _write_template(tmp_path, template_id, "  mood: [calm]\n")
    results = recommend_templates(moods=["calm", "unknown"], project_root=tmp_path)
    assert [item.template.id for item in results] == ["four", "one", "three"]
    assert all(item.matches == ("mood:calm",) for item in results)


def test_recommendations_require_tags_and_validate_limit():
    with pytest.raises(ValueError, match="at least one"):
        recommend_templates()
    with pytest.raises(ValueError, match="between 1 and 3"):
        recommend_templates(moods=["calm"], limit=4)
