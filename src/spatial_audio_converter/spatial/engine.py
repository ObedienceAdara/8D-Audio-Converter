from __future__ import annotations

import numpy as np

from ..domain.models import AudioBuffer
from .hrtf import HRTFConvolver
from .ild import InterauralLevelDifference
from .itd import InterauralTimeDifference
from .panning import EqualPowerPanner
from .trajectory import TrajectoryGenerator


class SpatialEngine:
    """Compose spatial motion, legacy stereo mode, and binaural HRTF rendering."""

    def __init__(self, block_size: int = 1024, hrtf_ir_length: int = 256, hrtf_path: str | None = None) -> None:
        self.trajectory = TrajectoryGenerator()
        self.panner = EqualPowerPanner()
        self.ild = InterauralLevelDifference()
        self.itd = InterauralTimeDifference()
        self.hrtf = HRTFConvolver(ir_length=hrtf_ir_length, block_size=block_size, hrtf_path=hrtf_path)

    @property
    def hrtf_source(self) -> str:
        return self.hrtf.source

    def process(
        self,
        audio: AudioBuffer,
        speed_hz: float,
        depth: float,
        use_hrtf: bool = False,
        headphone_mode: bool = True,
    ) -> AudioBuffer:
        mono = audio.samples[:, 0] if audio.channels == 1 else np.mean(audio.samples[:, :2], axis=1)
        azimuth = self.trajectory.generate(audio.frames, audio.sample_rate, speed_hz, depth)

        if use_hrtf or headphone_mode:
            output = self.hrtf.process(mono, azimuth, audio.sample_rate)
        else:
            stereo = self.panner.pan(mono, azimuth)
            left_gain, right_gain = self.ild.gains(azimuth)
            stereo[:, 0] *= left_gain
            stereo[:, 1] *= right_gain
            output = self.itd.apply(stereo, azimuth, audio.sample_rate)

        return AudioBuffer(output, audio.sample_rate, audio.metadata)
