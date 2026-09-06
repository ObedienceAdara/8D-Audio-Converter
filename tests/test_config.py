import pytest

from spatial_audio_converter.config import AudioProcessingConfig


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pan_speed_hz", 0),
        ("pan_speed_hz", 2.1),
        ("depth", -0.01),
        ("depth", 1.01),
        ("reverb_delay_ms", 0),
        ("reverb_delay_ms", 501),
        ("reverb_decay", -0.01),
        ("reverb_decay", 1.01),
        ("reverb_mix", -0.01),
        ("reverb_mix", 1.01),
        ("target_rms_db", -60.1),
        ("target_rms_db", 0.1),
        ("limiter_db", -20.1),
        ("limiter_db", 0.1),
        ("output_format", "aac"),
        ("max_duration_seconds", 0),
    ],
)
def test_invalid_processing_config_is_rejected(field, value):
    with pytest.raises(ValueError):
        AudioProcessingConfig(**{field: value})


def test_boundary_values_are_accepted():
    config = AudioProcessingConfig(
        pan_speed_hz=2.0,
        depth=0.0,
        reverb_delay_ms=1,
        reverb_decay=1.0,
        reverb_mix=0.0,
        target_rms_db=-60.0,
        limiter_db=-20.0,
        output_format="wav",
        max_duration_seconds=1,
    )
    assert config.output_format == "wav"


def test_phase_two_hrtf_is_opt_in():
    assert AudioProcessingConfig().hrtf_enabled is False
