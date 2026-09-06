import numpy as np

from spatial_audio_converter.processing.loudness import LoudnessController


def test_loudness_targets_requested_rms_for_unsaturated_signal():
    source = np.ones((1000, 2), dtype=np.float32) * 0.05
    output = LoudnessController().process(source, target_rms_db=-18.0, limiter_db=-1.0)
    rms = float(np.sqrt(np.mean(np.square(output))))
    target = 10 ** (-18.0 / 20.0)
    assert np.isclose(rms, target, rtol=1e-4, atol=1e-5)


def test_loudness_never_exceeds_peak_ceiling():
    source = np.ones((1000, 2), dtype=np.float32) * 0.8
    output = LoudnessController().process(source, target_rms_db=-6.0, limiter_db=-1.0)
    ceiling = 10 ** (-1.0 / 20.0)
    assert float(np.max(np.abs(output))) <= ceiling + 1e-6


def test_loudness_preserves_silence():
    source = np.zeros((100, 2), dtype=np.float32)
    output = LoudnessController().process(source, target_rms_db=-18.0, limiter_db=-1.0)
    assert np.array_equal(output, source)
