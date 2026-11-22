from pathlib import Path

import pytest

from video_edit_agent.editing.review_workspace import generate_review_workspace


def test_workspace_links_existing_review_pages_and_lists_artifacts(tmp_path: Path):
    artifact_dir = tmp_path / ".video-agent"
    timeline_dir = artifact_dir / "timeline-visual-review"
    timeline_dir.mkdir(parents=True)
    (artifact_dir / "storyboard.html").write_text("<html></html>", encoding="utf-8")
    (timeline_dir / "review.html").write_text("<html></html>", encoding="utf-8")
    (artifact_dir / "draft-critique.json").write_text("{}", encoding="utf-8")

    page = generate_review_workspace(tmp_path)
    content = page.read_text(encoding="utf-8")

    assert page == artifact_dir / "review-workspace.html"
    assert 'href="storyboard.html"' in content
    assert 'href="timeline-visual-review/review.html"' in content
    assert "Rendered-draft critique" in content
    assert "Available" in content
    assert "does not approve clips" in content


def test_workspace_supports_nested_output_path(tmp_path: Path):
    (tmp_path / ".video-agent").mkdir()
    page = generate_review_workspace(tmp_path, output_path=tmp_path / "reports" / "workspace.html")

    assert page.is_file()
    assert "../.video-agent/storyboard.html" in page.read_text(encoding="utf-8")


def test_workspace_rejects_output_outside_project(tmp_path: Path):
    with pytest.raises(ValueError, match="inside the project folder"):
        generate_review_workspace(tmp_path, output_path=tmp_path.parent / "workspace.html")
