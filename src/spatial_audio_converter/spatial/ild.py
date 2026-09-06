from __future__ import annotations

import numpy as np


class InterauralLevelDifference:
    """Simple frequency-independent ILD model for a compact binaural renderer.

    Positive azimuth means the source is to the listener's right. The far ear
    is attenuated increasingly with azimuth magnitude; the near ear remains at
    unity gain. This is an explicit perceptual model, not a measured HRTF.
    """

    def __init__(self, max_far_ear_attenuation_db: float = 6.0) -> None:
        if max_far_ear_attenuation_db < 0:
            raise ValueError("max_far_ear_attenuation_db must be non-negative")
        self.max_far_ear_attenuation_db = float(max_far_ear_attenuation_db)

    def gains(self, azimuth_deg: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        azimuth = np.clip(np.asarray(azimuth_deg, dtype=np.float32), -90.0, 90.0)
        factor = np.sin(np.deg2rad(np.abs(azimuth)))
        attenuation_db = self.max_far_ear_attenuation_db * factor
        far_gain = np.power(10.0, -attenuation_db / 20.0).astype(np.float32)
        left = np.where(azimuth > 0.0, far_gain, 1.0).astype(np.float32)
        right = np.where(azimuth < 0.0, far_gain, 1.0).astype(np.float32)
        return left, right
