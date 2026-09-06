from __future__ import annotations

import numpy as np

from ..domain.models import AudioBuffer
from .hrtf import HRTFConvolver
from .ild import InterauralLevelDifference
from .itd import InterauralTimeDifference
from .panning import EqualPowerPanner
from .trajectory import TrajectoryGenerator


class SpatialEngine:
    """Compose trajectory, panning, ILD/ITD, and optional analytic HRTF."""

    def __init__(self, block_size: int = 4096, hrtf_ir_length: int = 96) -> None:
        self.trajectory = TrajectoryGenerator()
        self.panner = EqualPowerPanner()
        self.ild = InterauralLevelDifference()
        self.itd = InterauralTimeDifference()
        self.hrtf = HRTFConvolver(ir_length=hrtf_ir_length, block_size=block_size, ild=self.ild, itd=self.itd)

    def process(
        self,
        audio: AudioBuffer,
        speed_hz: float,
        depth: float,
        use_hrtf: bool = False,
    ) -> AudioBuffer:
        if audio.channels == 1:
            mono = audio.samples[:, 0]
        else:
            mono = np.mean(audio.samples[:, :2], axis=1)
        azimuth = self.trajectory.generate(audio.frames, audio.sample_rate, speed_hz, depth)

        if use_hrtf:
            output = self.hrtf.process(mono, azimuth, audio.sample_rate)
        else:
            stereo = self.panner.pan(mono, azimuth)
            left_gain, right_gain = self.ild.gains(azimuth)
            stereo[:, 0] *= left_gain
            stereo[:, 1] *= right_gain
            output = self.itd.apply(stereo, azimuth, audio.sample_rate)

        return AudioBuffer(output, audio.sample_rate, audio.metadata)
