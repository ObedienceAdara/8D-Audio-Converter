from __future__ import annotations

import numpy as np


class LoudnessController:
    """RMS-based loudness normalization followed by a hard peak ceiling."""

    def process(self, stereo: np.ndarray, target_rms_db: float, limiter_db: float) -> np.ndarray:
        x = np.asarray(stereo, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
        if rms > 1e-12:
            target_rms = 10.0 ** (target_rms_db / 20.0)
            x = x * (target_rms / rms)
        ceiling = 10.0 ** (limiter_db / 20.0)
        return np.clip(x, -ceiling, ceiling).astype(np.float32)
