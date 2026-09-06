from __future__ import annotations

import numpy as np


class EqualPowerPanner:
    """Convert a mono source and azimuth trajectory into a stereo field."""

    def pan(self, source: np.ndarray, azimuth_deg: np.ndarray) -> np.ndarray:
        source = np.asarray(source, dtype=np.float32)
        azimuth = np.clip(np.asarray(azimuth_deg, dtype=np.float32), -90.0, 90.0)
        theta = np.deg2rad(azimuth + 90.0)
        left_gain = np.cos(theta / 2.0)
        right_gain = np.sin(theta / 2.0)
        left = source * left_gain
        right = source * right_gain
        return np.column_stack((left, right)).astype(np.float32)
