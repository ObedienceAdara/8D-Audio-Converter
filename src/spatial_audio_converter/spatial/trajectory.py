from __future__ import annotations

import numpy as np


class TrajectoryGenerator:
    """Generate a smooth azimuth trajectory in degrees."""

    def generate(self, frames: int, sample_rate: int, speed_hz: float, depth: float) -> np.ndarray:
        if frames <= 0:
            return np.empty(0, dtype=np.float32)
        t = np.arange(frames, dtype=np.float64) / sample_rate
        # One complete left-right sweep follows the requested oscillator speed.
        return (90.0 * depth * np.sin(2.0 * np.pi * speed_hz * t)).astype(np.float32)
