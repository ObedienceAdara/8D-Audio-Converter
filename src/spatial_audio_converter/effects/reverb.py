from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


class RoomReverb:
    """Lightweight multi-tap room model based on early reflections and decay."""

    def process(self, stereo: np.ndarray, sample_rate: int, delay_ms: int, decay: float, mix: float) -> np.ndarray:
        if mix <= 0 or decay <= 0:
            return stereo.astype(np.float32, copy=True)
        max_delay = max(1, int(sample_rate * delay_ms / 1000))
        delays = [max(1, max_delay), max(2, int(max_delay * 1.73)), max(3, int(max_delay * 2.61))]
        gains = [decay, decay * 0.65, decay * 0.4]
        ir = np.zeros(delays[-1] + 1, dtype=np.float32)
        ir[0] = 1.0
        for d, g in zip(delays, gains):
            ir[d] += float(g)
        wet = np.empty_like(stereo, dtype=np.float32)
        for channel in range(stereo.shape[1]):
            wet[:, channel] = lfilter(ir, [1.0], stereo[:, channel]).astype(np.float32)
        return ((1.0 - mix) * stereo + mix * wet).astype(np.float32)
