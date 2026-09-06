from __future__ import annotations

import numpy as np

from ..domain.models import AudioBuffer
from .hrtf import HRTFConvolver
from .panning import EqualPowerPanner
from .trajectory import TrajectoryGenerator


class SpatialEngine:
    """Compose trajectory, equal-power panning, and optional binaural HRTF filtering."""

    def __init__(self, block_size: int = 4096, hrtf_ir_length: int = 96) -> None:
        self.trajectory = TrajectoryGenerator()
        self.panner = EqualPowerPanner()
        self.hrtf = HRTFConvolver(ir_length=hrtf_ir_length, block_size=block_size)

    def process(self, audio: AudioBuffer, speed_hz: float, depth: float, use_hrtf: bool = True) -> AudioBuffer:
        if audio.channels == 1:
            mono = audio.samples[:, 0]
        else:
            mono = np.mean(audio.samples[:, :2], axis=1)
        azimuth = self.trajectory.generate(audio.frames, audio.sample_rate, speed_hz, depth)

        # Equal-power panning remains the lightweight fallback and a useful
        # reference implementation for tests/benchmarks. The HRTF path replaces
        # it with ear-specific delay/shaping when binaural rendering is enabled.
        if use_hrtf:
            output = self.hrtf.process(mono, azimuth, audio.sample_rate)
        else:
            output = self.panner.pan(mono, azimuth)
        return AudioBuffer(output, audio.sample_rate, audio.metadata)
