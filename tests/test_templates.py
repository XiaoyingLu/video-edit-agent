from pathlib import Path

import pytest
from pydantic import ValidationError

from video_edit_agent.templates import TemplateSpec, load_template


def _write_template(directory: Path, template_id: str, body: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{template_id}.yaml").write_text(body, encoding="utf-8")


def test_builtin_templates_load_and_resolve_inheritance(tmp_path: Path):
    for template_id in ("cinematic", "vlog", "travel-montage", "social"):
        spec, source = load_template(template_id, project_root=tmp_path, home_dir=tmp_path / "home")
        assert spec.id == template_id
        assert source.is_file()
        assert spec.title_styles["opening"].placement in {"top", "center", "bottom"}
    travel, _ = load_template("travel-montage", project_root=tmp_path, home_dir=tmp_path / "home")
    assert travel.music.volume == 0.25
    cinematic, _ = load_template("cinematic", project_root=tmp_path, home_dir=tmp_path / "home")
    assert cinematic.music.loudness_normalization is True
    assert cinematic.music.target_lufs == -16.0
    assert travel.title_styles["minimal"].size == 0.72


def test_template_schema_rejects_unknown_fields_and_raw_filter_strings():
    with pytest.raises(ValidationError):
        TemplateSpec.model_validate({"id": "safe", "version": 1, "ffmpeg": "drawtext=..."})
    with pytest.raises(ValidationError):
        TemplateSpec.model_validate(
            {
                "id": "safe",
                "version": 1,
                "title_styles": {"opening": {"animation": "sendcmd"}},
            }
        )


def test_project_template_overrides_home_and_builtin(tmp_path: Path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    _write_template(
        home / ".video-agent" / "templates",
        "cinematic",
        """
id: cinematic
version: 1
music:
  volume: 0.11
""",
    )
    _write_template(
        project / ".video-agent" / "templates",
        "cinematic",
        """
id: cinematic
version: 1
music:
  volume: 0.12
""",
    )
    spec, source = load_template("cinematic", project, home_dir=home)
    assert spec.music.volume == 0.12
    assert source.parent == project / ".video-agent" / "templates"


def test_template_inheritance_rejects_cycles(tmp_path: Path):
    library = tmp_path / "library"
    _write_template(library, "one", "id: one\nversion: 1\nextends: two\n")
    _write_template(library, "two", "id: two\nversion: 1\nextends: one\n")
    with pytest.raises(ValueError, match="inheritance cycle"):
        load_template("one", home_dir=tmp_path / "home", library_dir=library)


def test_template_assets_must_stay_inside_template_folder(tmp_path: Path):
    library = tmp_path / "library"
    _write_template(
        library,
        "assets",
        """
id: assets
version: 1
assets:
  - ../outside.webm
""",
    )
    with pytest.raises(ValueError, match="escapes its template folder"):
        load_template("assets", home_dir=tmp_path / "home", library_dir=library)


def test_template_assets_must_exist(tmp_path: Path):
    library = tmp_path / "library"
    _write_template(
        library,
        "assets",
        """
id: assets
version: 1
assets:
  - overlays/logo.webm
""",
    )
    with pytest.raises(FileNotFoundError, match="template asset does not exist"):
        load_template("assets", home_dir=tmp_path / "home", library_dir=library)


def test_template_subtitle_defaults_and_custom_settings():
    defaults = TemplateSpec.model_validate({"id": "plain", "version": 1})
    assert defaults.subtitles.style == "classic"
    assert defaults.subtitles.position == "bottom"
    assert defaults.subtitles.font_size == 28

    custom = TemplateSpec.model_validate(
        {
            "id": "captions",
            "version": 1,
            "subtitles": {"style": "boxed", "position": "center", "font_size": 36},
        }
    )
    assert custom.subtitles.style == "boxed"
    assert custom.subtitles.position == "center"
    assert custom.subtitles.font_size == 36


@pytest.mark.parametrize(
    "subtitles",
    [
        {"style": "neon"},
        {"position": "left"},
        {"font_size": 90},
        {"force_style": "FontName=Bad"},
    ],
)
def test_template_subtitle_schema_rejects_unsupported_values(subtitles):
    with pytest.raises(ValidationError):
        TemplateSpec.model_validate({"id": "unsafe", "version": 1, "subtitles": subtitles})
