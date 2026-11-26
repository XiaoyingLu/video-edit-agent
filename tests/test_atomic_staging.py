import pytest

from video_edit_agent.rendering.atomic_staging import RunWorkspace


def test_run_workspace_checksums_and_promotes_only_verified_files(tmp_path):
    destination = tmp_path / ".video-agent" / "render.mp4"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"old-render")

    with RunWorkspace(tmp_path) as workspace:
        staged = workspace.stage_path("renders/draft.mp4")
        staged.write_bytes(b"validated-render")
        checksum = workspace.checksum(staged)
        assert len(checksum) == 64
        assert destination.read_bytes() == b"old-render"

        workspace.promote(staged, destination)
        assert destination.read_bytes() == b"validated-render"
        assert (workspace.path / "checksums.json").is_file()

    assert not list((tmp_path / ".video-agent" / "workspaces").iterdir())


def test_run_workspace_rejects_modified_stage_and_preserves_active_output(tmp_path):
    destination = tmp_path / "render.mp4"
    destination.write_bytes(b"previous")

    with RunWorkspace(tmp_path) as workspace:
        staged = workspace.stage_path("clip.mp4")
        staged.write_bytes(b"validated")
        workspace.checksum(staged)
        staged.write_bytes(b"tampered")
        with pytest.raises(ValueError, match="checksum mismatch"):
            workspace.promote(staged, destination)

    assert destination.read_bytes() == b"previous"


def test_run_workspace_rolls_back_when_manifest_commit_fails(tmp_path):
    destination = tmp_path / "render.mp4"
    destination.write_bytes(b"previous")

    def fail_commit():
        raise RuntimeError("manifest write failed")

    with RunWorkspace(tmp_path) as workspace:
        staged = workspace.stage_path("render.mp4")
        staged.write_bytes(b"new")
        workspace.checksum(staged)
        with pytest.raises(RuntimeError, match="manifest write failed"):
            workspace.promote(staged, destination, finalize=fail_commit)

    assert destination.read_bytes() == b"previous"


def test_run_workspace_rejects_path_traversal(tmp_path):
    with RunWorkspace(tmp_path) as workspace:
        with pytest.raises(ValueError, match="inside the run workspace"):
            workspace.stage_path("../escape.mp4")
