import numpy as np
import pytest

from spatial_audio_converter.analysis.waveform import summarize_waveform


def test_waveform_returns_peak_envelope():
    samples = np.array([0.0, 0.25, -0.5, 0.75, -1.0, 0.5], dtype=np.float32)
    envelope = summarize_waveform(samples, points=3)
    assert envelope == pytest.approx([0.25, 0.75, 1.0])


def test_waveform_accepts_stereo_and_empty_input():
    stereo = np.array([[0.1, -0.2], [0.5, 0.4], [-0.8, 0.1]], dtype=np.float32)
    assert summarize_waveform(stereo, points=2) == pytest.approx([0.15, 0.45])
    assert summarize_waveform(np.array([], dtype=np.float32)) == []
