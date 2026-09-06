from __future__ import annotations

import numpy as np

from .ild import InterauralLevelDifference
from .itd import InterauralTimeDifference


class HRTFConvolver:
    """Self-contained analytic binaural approximation composed from ILD and ITD.

    This is an explicit, replaceable approximation boundary; it is not a
    measured HRTF dataset or a perceptually validated renderer.
    """

    def __init__(
        self,
        ir_length: int = 96,
        block_size: int = 4096,
        ild: InterauralLevelDifference | None = None,
        itd: InterauralTimeDifference | None = None,
    ) -> None:
        # Retain these knobs for compatibility with the architecture and future
        # measured-HRTF convolution. The current analytic model is delay/gain based.
        self.ir_length = max(16, ir_length)
        self.block_size = max(256, block_size)
        self.ild = ild or InterauralLevelDifference(max_far_ear_attenuation_db=6.0)
        self.itd = itd or InterauralTimeDifference(max_delay_seconds=0.00065)

    def process(self, mono: np.ndarray, azimuth_deg: np.ndarray, sample_rate: int) -> np.ndarray:
        source = np.asarray(mono, dtype=np.float32).reshape(-1)
        azimuth = np.asarray(azimuth_deg, dtype=np.float32).reshape(-1)
        if len(source) != len(azimuth):
            raise ValueError("Azimuth trajectory length must match source length.")
        left_gain, right_gain = self.ild.gains(azimuth)
        stereo = np.column_stack((source * left_gain, source * right_gain)).astype(np.float32)
        return self.itd.apply(stereo, azimuth, sample_rate)
