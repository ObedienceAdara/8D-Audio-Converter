import pytest

from spatial_audio_converter.presets import get_preset, list_presets


def test_all_presets_are_valid_configs():
    names = {item["name"] for item in list_presets()}
    assert names == {"balanced", "deep", "clean"}
    for name in names:
        config = get_preset(name)
        assert 0 < config.pan_speed_hz <= 2
        assert 0 <= config.depth <= 1


def test_preset_values_are_distinct_enough_to_change_motion():
    assert get_preset("balanced").depth != get_preset("deep").depth
    assert get_preset("clean").room_enabled is False


def test_unknown_preset_is_rejected():
    with pytest.raises(ValueError, match="Unknown preset"):
        get_preset("cinema")
