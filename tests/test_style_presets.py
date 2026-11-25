import pytest

from video_edit_agent.rendering.style_presets import STYLE_PRESETS, resolve_style_preset
from video_edit_agent.templates import load_builtin_template


def test_all_style_presets_have_valid_render_settings():
    assert set(STYLE_PRESETS) == {"cinematic", "vlog", "travel-montage", "social"}
    for preset in STYLE_PRESETS.values():
        assert preset.fit_mode in {"crop", "pad"}
        assert preset.music_mode in {"original", "music_only", "blend"}
        assert 0.01 <= preset.music_volume <= 2.0
        assert preset.music_fade_in >= 0
        assert preset.music_fade_out >= 0
        assert -24 <= preset.music_target_lufs <= -12
        assert -9 <= preset.music_true_peak_db <= -1
        assert 5 <= preset.music_loudness_range <= 20
        assert isinstance(preset.beat_punch_in, bool)


@pytest.mark.parametrize("name", ["cinematic", "vlog", "travel-montage", "social"])
def test_resolve_style_preset_returns_named_preset(name):
    assert resolve_style_preset(name) is STYLE_PRESETS[name]


@pytest.mark.parametrize("name", ["cinematic", "vlog", "travel-montage", "social"])
def test_legacy_preset_values_are_loaded_from_builtin_template(name):
    template, _ = load_builtin_template(name)
    preset = STYLE_PRESETS[name]
    assert preset.fit_mode == template.render.fit_mode
    assert preset.music_mode == template.music.mode
    assert preset.music_volume == template.music.volume
    assert preset.music_fade_in == template.music.fade_in
    assert preset.music_fade_out == template.music.fade_out
    assert preset.music_ducking == template.music.ducking
    assert preset.music_loudness_normalization == template.music.loudness_normalization
    assert preset.music_target_lufs == template.music.target_lufs
    assert preset.music_true_peak_db == template.music.true_peak_db
    assert preset.music_loudness_range == template.music.loudness_range
    assert preset.beat_punch_in == template.render.beat_punch_in


def test_resolve_style_preset_preserves_legacy_defaults_when_omitted():
    assert resolve_style_preset(None) is None


def test_resolve_style_preset_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown style preset"):
        resolve_style_preset("noir")


def test_beat_punch_in_is_opted_in_only_for_montage_and_social_presets():
    assert STYLE_PRESETS["cinematic"].beat_punch_in is False
    assert STYLE_PRESETS["vlog"].beat_punch_in is False
    assert STYLE_PRESETS["travel-montage"].beat_punch_in is True
    assert STYLE_PRESETS["social"].beat_punch_in is True
