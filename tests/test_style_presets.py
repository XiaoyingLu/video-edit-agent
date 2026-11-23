import pytest

from video_edit_agent.rendering.style_presets import STYLE_PRESETS, resolve_style_preset


def test_all_style_presets_have_valid_render_settings():
    assert set(STYLE_PRESETS) == {"cinematic", "vlog", "travel-montage", "social"}
    for preset in STYLE_PRESETS.values():
        assert preset.fit_mode in {"crop", "pad"}
        assert preset.music_mode in {"original", "music_only", "blend"}
        assert 0.01 <= preset.music_volume <= 2.0
        assert preset.music_fade_in >= 0
        assert preset.music_fade_out >= 0
        assert isinstance(preset.beat_punch_in, bool)


@pytest.mark.parametrize("name", ["cinematic", "vlog", "travel-montage", "social"])
def test_resolve_style_preset_returns_named_preset(name):
    assert resolve_style_preset(name) is STYLE_PRESETS[name]


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
