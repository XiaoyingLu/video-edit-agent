from pathlib import Path

import pytest
from typer.testing import CliRunner

from video_edit_agent.cli import app
from video_edit_agent.editing.timeline_schemas import DraftTimelineClip, ProjectDraftTimeline
from video_edit_agent.rendering.template_capture import save_template_from_project
from video_edit_agent.templates import load_template


def _project(tmp_path: Path, *, title_overlays: bool = True) -> Path:
    root = tmp_path / "project"
    metadata = root / ".video-agent"
    metadata.mkdir(parents=True)
    clips = [
        DraftTimelineClip(
            clip_id=f"clip-{index}",
            sequence_index=index,
            video_path=f"media/{index}.mp4",
            source_start_seconds=0,
            source_end_seconds=4,
            duration_seconds=4,
            combined_score=90,
            pacing_note="Good pacing",
            transition_in="cut" if index == 1 else "fadeblack",
            transition_duration_seconds=0 if index == 1 else 0.5,
            decision="approved",
        )
        for index in (1, 2)
    ]
    timeline = ProjectDraftTimeline(
        project_path=str(root),
        style="cinematic",
        pacing="moderate",
        target_duration_seconds=8,
        proposed_duration_seconds=7.5,
        approved_duration_seconds=7.5,
        proposed_clip_count=2,
        clips=clips,
        timeline_approved=True,
    )
    (metadata / "optimized-timeline.json").write_text(
        timeline.model_dump_json(indent=2), encoding="utf-8"
    )
    if title_overlays:
        (metadata / "title-overlays.json").write_text(
            f"""{{
              "schema_version": "1.1",
              "project_path": "{str(root)}",
              "source_timeline": ".video-agent/optimized-timeline.json",
              "overlays": [
                {{"overlay_id":"title","text":"Do not copy me","start_seconds":0,"end_seconds":2,
                 "placement":"center","style":"bold","animation":"fade",
                 "sequence_role":"opening","decision":"approved"}},
                {{"overlay_id":"logline","text":"Private wording","start_seconds":0,"end_seconds":2,
                 "placement":"bottom","style":"minimal","animation":"none",
                 "sequence_role":"opening","decision":"approved"}},
                {{"overlay_id":"closing","text":"Private ending","start_seconds":6,"end_seconds":8,
                 "placement":"bottom","style":"cinematic","animation":"fade",
                 "sequence_role":"closing","decision":"approved"}},
                {{"overlay_id":"pending","text":"Not approved","start_seconds":2,"end_seconds":4,
                 "placement":"top","style":"bold","sequence_role":"opening","decision":"pending"}}
              ]
            }}""",
            encoding="utf-8",
        )
    return root


def test_save_template_captures_approved_styles_and_transitions(tmp_path):
    root = _project(tmp_path)
    spec, destination, _ = save_template_from_project(
        root, "my-look", output_path=tmp_path / "my-look.yaml"
    )

    assert destination.is_file()
    assert spec.transitions.default == "fadeblack"
    assert spec.title_styles["opening-1"].placement == "center"
    assert spec.title_styles["opening-1"].size == 1.2
    assert spec.title_styles["opening-2"].placement == "bottom"
    assert [item.slot for item in spec.sequences["opening"]] == ["{title}", "{logline}"]
    assert spec.sequences["closing"][0].slot == "{closing_line}"
    assert "Private wording" not in destination.read_text(encoding="utf-8")


def test_save_template_uses_profile_defaults_without_overlay_document(tmp_path):
    root = _project(tmp_path, title_overlays=False)
    spec, _, notes = save_template_from_project(
        root, "profile-look", output_path=tmp_path / "profile-look.yaml"
    )

    assert spec.music.volume == load_template("cinematic")[0].music.volume
    assert spec.render.fit_mode == "crop"
    assert spec.sequences["opening"][0].slot == "{title}"
    assert any("No title-overlay document" in note for note in notes)


def test_save_template_requires_approved_timeline(tmp_path):
    root = _project(tmp_path)
    path = root / ".video-agent" / "optimized-timeline.json"
    data = ProjectDraftTimeline.model_validate_json(path.read_text(encoding="utf-8"))
    data.timeline_approved = False
    path.write_text(data.model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="must be approved"):
        save_template_from_project(root, "not-approved", output_path=tmp_path / "x.yaml")


def test_save_template_refuses_overwrite(tmp_path):
    root = _project(tmp_path)
    output = tmp_path / "existing.yaml"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        save_template_from_project(root, "existing", output_path=output)


def test_save_template_supports_music_overrides(tmp_path):
    root = _project(tmp_path, title_overlays=False)
    spec, _, _ = save_template_from_project(
        root,
        "custom-mix",
        output_path=tmp_path / "custom-mix.yaml",
        music_volume=0.31,
        music_fade_in=2.0,
        music_ducking=False,
    )

    assert spec.music.volume == 0.31
    assert spec.music.fade_in == 2.0
    assert spec.music.ducking is False


def test_saved_template_can_be_loaded_by_template_library(tmp_path):
    root = _project(tmp_path, title_overlays=False)
    output = tmp_path / ".video-agent" / "templates" / "saved.yaml"
    save_template_from_project(root, "saved", output_path=output)

    loaded, source = load_template("saved", project_root=tmp_path, home_dir=tmp_path / "home")
    assert loaded.id == "saved"
    assert source == output.resolve()


def test_template_save_cli_requires_from_project_flag(tmp_path):
    result = CliRunner().invoke(app, ["template", "save", str(tmp_path), "example"])
    assert result.exit_code == 1
    assert "Specify --from-project" in result.output


def test_template_save_cli_reports_saved_template(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import video_edit_agent.cli.templates as template_cli

    captured = {}

    def fake_save(folder, template_id, **kwargs):
        captured["folder"] = folder
        captured["template_id"] = template_id
        captured.update(kwargs)
        template = SimpleNamespace(
            id=template_id,
            version=1,
            title_styles={"opening": object()},
            sequences={"opening": [object()]},
            transitions=SimpleNamespace(default="crossfade"),
            music=SimpleNamespace(
                mode="blend",
                volume=0.3,
                fade_in=1.0,
                fade_out=2.0,
                ducking=False,
            ),
        )
        return template, tmp_path / "example.yaml", []

    monkeypatch.setattr(template_cli, "save_template_from_project", fake_save)
    result = CliRunner().invoke(
        app,
        [
            "template",
            "save",
            str(tmp_path),
            "example",
            "--from-project",
            "--music-volume",
            "0.3",
            "--no-music-ducking",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["folder"] == tmp_path
    assert captured["template_id"] == "example"
    assert captured["music_volume"] == 0.3
    assert captured["music_ducking"] is False
    assert "No footage" in result.output
