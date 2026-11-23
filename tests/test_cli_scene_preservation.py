import json
from pathlib import Path

from typer.testing import CliRunner

from video_edit_agent.cli import app
from video_edit_agent.editing.schemas import (
    EditReviewItem,
    ProjectEditProposal,
    VideoEditProposal,
)


def test_review_edits_displays_scene_preservation_hint(tmp_path: Path) -> None:
    artifact_dir = tmp_path / ".video-agent"
    artifact_dir.mkdir()
    (artifact_dir / "silence-analysis.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "understanding.json").write_text("{}", encoding="utf-8")
    proposal = ProjectEditProposal(
        project_path=str(tmp_path.resolve()),
        source_artifacts=[
            ".video-agent/silence-analysis.json",
            ".video-agent/understanding.json",
        ],
        video_count=1,
        review_item_count=1,
        videos=[
            VideoEditProposal(
                path="clip.mp4",
                duration_seconds=10,
                review_items=[
                    EditReviewItem(
                        item_id="clip-review-001",
                        start_seconds=3,
                        end_seconds=7,
                        duration_seconds=4,
                        suggested_action="preserve",
                        review_reason="Suggested action: preserve (scenic_breath).",
                        scene_role="scenic_breath",
                        scene_id="scene-1",
                        scene_summary="A quiet mountain panorama",
                    )
                ],
            )
        ],
    )
    (artifact_dir / "edit-proposal.json").write_text(proposal.model_dump_json(), encoding="utf-8")

    result = CliRunner().invoke(app, ["review-edits", str(tmp_path)], input="s\n")

    assert result.exit_code == 0, result.output
    assert "Suggested action: preserve" in result.output
    assert "0 approved, 0 rejected, 1 pending" in result.output
    saved = json.loads((artifact_dir / "edit-proposal.json").read_text(encoding="utf-8"))
    assert saved["videos"][0]["review_items"][0]["decision"] == "pending"
