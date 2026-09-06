import numpy as np

from spatial_audio_converter.config import AudioProcessingConfig
from spatial_audio_converter.domain.models import AudioBuffer
from spatial_audio_converter.processing.loudness import LoudnessController
from spatial_audio_converter.spatial.hrtf import HRTFConvolver
from spatial_audio_converter.spatial.panning import EqualPowerPanner
from spatial_audio_converter.spatial.trajectory import TrajectoryGenerator


def test_config_rejects_invalid_depth():
    try:
        AudioProcessingConfig(depth=1.5)
    except ValueError:
        return
    raise AssertionError("invalid depth was accepted")


def test_trajectory_stays_inside_expected_azimuth_range():
    trajectory = TrajectoryGenerator().generate(44100, 44100, 0.5, 1.0)
    assert trajectory.min() >= -90.0
    assert trajectory.max() <= 90.0


def test_equal_power_panner_outputs_stereo():
    source = np.ones(128, dtype=np.float32)
    azimuth = np.linspace(-90, 90, 128, dtype=np.float32)
    output = EqualPowerPanner().pan(source, azimuth)
    assert output.shape == (128, 2)
    assert np.isfinite(output).all()


def test_hrtf_convolver_preserves_frame_count():
    source = np.zeros(512, dtype=np.float32)
    azimuth = np.zeros(512, dtype=np.float32)
    output = HRTFConvolver().process(source, azimuth, 44100)
    assert output.shape == (512, 2)


def test_loudness_controller_respects_ceiling():
    source = np.ones((1000, 2), dtype=np.float32) * 0.8
    output = LoudnessController().process(source, -18, -1)
    assert float(np.max(np.abs(output))) <= 10 ** (-1 / 20) + 1e-6


def test_audio_buffer_metadata_and_duration():
    buffer = AudioBuffer(np.zeros((44100, 2), dtype=np.float32), 44100)
    assert buffer.channels == 2
    assert buffer.duration_seconds == 1.0
