from __future__ import annotations

import numpy as np


class EqualPowerPanner:
    """Map source azimuth to constant-power stereo gains.

    Azimuth is defined as -90° = hard left, 0° = center, and +90° = hard right.
    The gain pair satisfies L² + R² = 1 for every azimuth, avoiding the level dip
    that a linear amplitude crossfade introduces around the center.
    """

    def gains(self, azimuth_deg: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        azimuth = np.clip(np.asarray(azimuth_deg, dtype=np.float32), -90.0, 90.0)
        theta = np.deg2rad(azimuth + 90.0)
        left_gain = np.cos(theta / 2.0).astype(np.float32)
        right_gain = np.sin(theta / 2.0).astype(np.float32)
        return left_gain, right_gain

    def pan(self, source: np.ndarray, azimuth_deg: np.ndarray | float) -> np.ndarray:
        signal = np.asarray(source, dtype=np.float32).reshape(-1)
        left_gain, right_gain = self.gains(azimuth_deg)
        if np.ndim(left_gain) == 0:
            left_gain = np.full(len(signal), float(left_gain), dtype=np.float32)
            right_gain = np.full(len(signal), float(right_gain), dtype=np.float32)
        elif len(left_gain) != len(signal):
            raise ValueError("Azimuth trajectory length must match source length.")
        return np.column_stack((signal * left_gain, signal * right_gain)).astype(np.float32)
