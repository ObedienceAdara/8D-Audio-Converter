from __future__ import annotations

import numpy as np


class InterauralTimeDifference:
    """Time-varying integer-sample ITD model.

    Positive azimuth means source-right, so the left ear receives the delayed
    path. Negative azimuth means source-left, so the right ear is delayed.
    """

    def __init__(self, max_delay_seconds: float = 0.00065) -> None:
        if max_delay_seconds < 0:
            raise ValueError("max_delay_seconds must be non-negative")
        self.max_delay_seconds = float(max_delay_seconds)

    def delays(self, azimuth_deg: np.ndarray | float, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        azimuth = np.clip(np.asarray(azimuth_deg, dtype=np.float32), -90.0, 90.0)
        delay_seconds = self.max_delay_seconds * np.sin(np.deg2rad(np.abs(azimuth)))
        delay_samples = np.rint(delay_seconds * sample_rate).astype(np.int32)
        left = np.where(azimuth > 0.0, delay_samples, 0).astype(np.int32)
        right = np.where(azimuth < 0.0, delay_samples, 0).astype(np.int32)
        return left, right

    @staticmethod
    def _apply_time_varying_delay(signal: np.ndarray, delays: np.ndarray) -> np.ndarray:
        source = np.asarray(signal, dtype=np.float32).reshape(-1)
        delay = np.asarray(delays, dtype=np.int32).reshape(-1)
        if len(source) != len(delay):
            raise ValueError("Delay trajectory length must match signal length.")
        indices = np.arange(len(source), dtype=np.int64) - delay.astype(np.int64)
        valid = indices >= 0
        output = np.zeros_like(source)
        output[valid] = source[indices[valid]]
        return output

    def apply(
        self,
        stereo: np.ndarray,
        azimuth_deg: np.ndarray | float,
        sample_rate: int,
    ) -> np.ndarray:
        audio = np.asarray(stereo, dtype=np.float32)
        if audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError("ITD processing requires stereo audio with shape (frames, 2).")
        left_delay, right_delay = self.delays(azimuth_deg, sample_rate)
        if np.ndim(left_delay) == 0:
            left_delay = np.full(len(audio), int(left_delay), dtype=np.int32)
            right_delay = np.full(len(audio), int(right_delay), dtype=np.int32)
        elif len(left_delay) != len(audio):
            raise ValueError("Azimuth trajectory length must match stereo frame count.")
        left = self._apply_time_varying_delay(audio[:, 0], left_delay)
        right = self._apply_time_varying_delay(audio[:, 1], right_delay)
        return np.column_stack((left, right)).astype(np.float32)
