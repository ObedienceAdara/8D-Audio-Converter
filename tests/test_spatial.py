import numpy as np

from spatial_audio_converter.domain.models import AudioBuffer
from spatial_audio_converter.spatial.engine import SpatialEngine
from spatial_audio_converter.spatial.hrtf import HRTFConvolver
from spatial_audio_converter.spatial.ild import InterauralLevelDifference
from spatial_audio_converter.spatial.itd import InterauralTimeDifference
from spatial_audio_converter.spatial.panning import EqualPowerPanner
from spatial_audio_converter.spatial.trajectory import TrajectoryGenerator


def test_trajectory_stays_inside_expected_azimuth_range():
    trajectory = TrajectoryGenerator().generate(44100, 44100, 0.5, 1.0)
    assert trajectory.min() >= -90.0
    assert trajectory.max() <= 90.0


def test_trajectory_zero_depth_is_centered():
    trajectory = TrajectoryGenerator().generate(1024, 44100, 0.5, 0.0)
    assert np.allclose(trajectory, 0.0)


def test_trajectory_completes_expected_number_of_cycles():
    sample_rate = 1000
    speed_hz = 2.0
    trajectory = TrajectoryGenerator().generate(sample_rate, sample_rate, speed_hz, 1.0)
    assert np.isclose(trajectory[0], 0.0, atol=1e-6)
    assert np.isclose(trajectory[-1], 0.0, atol=1e-6)


def test_equal_power_panner_is_constant_power():
    azimuth = np.linspace(-90, 90, 181, dtype=np.float32)
    left, right = EqualPowerPanner().gains(azimuth)
    assert np.allclose(left**2 + right**2, 1.0, atol=1e-6)
    assert np.isclose(left[0], 1.0, atol=1e-6)
    assert np.isclose(right[0], 0.0, atol=1e-6)
    assert np.isclose(left[-1], 0.0, atol=1e-6)
    assert np.isclose(right[-1], 1.0, atol=1e-6)


def test_equal_power_panner_requires_matching_trajectory_length():
    source = np.ones(128, dtype=np.float32)
    azimuth = np.zeros(127, dtype=np.float32)
    try:
        EqualPowerPanner().pan(source, azimuth)
    except ValueError:
        return
    raise AssertionError("mismatched trajectory length was accepted")


def test_ild_is_neutral_at_center_and_attenuates_far_ear():
    ild = InterauralLevelDifference(max_far_ear_attenuation_db=6.0)
    left, right = ild.gains(np.array([-90.0, 0.0, 90.0], dtype=np.float32))
    assert np.isclose(left[0], 1.0)
    assert np.isclose(right[0], 10 ** (-6 / 20), atol=1e-6)
    assert np.isclose(left[1], 1.0)
    assert np.isclose(right[1], 1.0)
    assert np.isclose(left[2], 10 ** (-6 / 20), atol=1e-6)
    assert np.isclose(right[2], 1.0)


def test_ild_is_monotonic_with_azimuth_magnitude():
    azimuth = np.array([0.0, 30.0, 60.0, 90.0], dtype=np.float32)
    left, _ = InterauralLevelDifference().gains(azimuth)
    assert np.all(np.diff(left) <= 1e-7)


def test_itd_direction_matches_source_side():
    itd = InterauralTimeDifference(max_delay_seconds=0.001)
    left_delay, right_delay = itd.delays(np.array([-90.0, 0.0, 90.0]), 1000)
    assert np.array_equal(left_delay, np.array([0, 0, 1], dtype=np.int32))
    assert np.array_equal(right_delay, np.array([1, 0, 0], dtype=np.int32))


def test_itd_applies_time_varying_delay_per_frame():
    source = np.zeros((5, 2), dtype=np.float32)
    source[0] = 1.0
    azimuth = np.array([90, 90, 90, 0, 0], dtype=np.float32)
    output = InterauralTimeDifference(max_delay_seconds=0.001).apply(source, azimuth, 1000)
    assert np.array_equal(output[:, 0], np.array([0, 1, 0, 0, 0], dtype=np.float32))
    assert np.array_equal(output[:, 1], np.array([1, 0, 0, 0, 0], dtype=np.float32))


def test_spatial_engine_zero_depth_is_centered():
    source = np.ones((256, 1), dtype=np.float32) * 0.1
    output = SpatialEngine().process(AudioBuffer(source, 48000), 0.5, 0.0, use_hrtf=False)
    assert np.allclose(output.samples[:, 0], output.samples[:, 1], atol=1e-7)


def test_spatial_engine_hard_right_is_right_dominant():
    source = np.ones((256, 1), dtype=np.float32) * 0.1
    engine = SpatialEngine()
    engine.trajectory.generate = lambda frames, sample_rate, speed_hz, depth: np.full(
        frames, 90.0, dtype=np.float32
    )
    output = engine.process(AudioBuffer(source, 48000), 0.5, 1.0, use_hrtf=False)
    assert float(np.mean(output.samples[:, 1])) > float(np.mean(output.samples[:, 0]))


def test_spatial_engine_produces_stereo_and_finite_samples():
    t = np.arange(2048, dtype=np.float32) / 48000.0
    source = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    audio = AudioBuffer(source[:, None], 48000)
    output = SpatialEngine().process(audio, speed_hz=0.5, depth=1.0, use_hrtf=False)
    assert output.samples.shape == (2048, 2)
    assert np.isfinite(output.samples).all()


def test_analytic_hrtf_preserves_frame_count_and_stereo_shape():
    source = np.zeros(512, dtype=np.float32)
    azimuth = np.zeros(512, dtype=np.float32)
    output = HRTFConvolver().process(source, azimuth, 44100)
    assert output.shape == (512, 2)
    assert np.isfinite(output).all()
