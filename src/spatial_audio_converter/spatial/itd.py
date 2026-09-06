from __future__ import annotations

import numpy as np


class InterauralTimeDifference:
    """Integer-sample ITD model with a bounded maximum acoustic delay."""

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
    def _delay_channel(signal: np.ndarray, delay_samples: int) -> np.ndarray:
        signal = np.asarray(signal, dtype=np.float32).reshape(-1)
        if delay_samples <= 0:
            return signal.copy()
        output = np.zeros_like(signal)
        if delay_samples < len(signal):
            output[delay_samples:] = signal[:-delay_samples]
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
            left_samples = int(left_delay)
            right_samples = int(right_delay)
        else:
            left_samples = int(np.max(left_delay))
            right_samples = int(np.max(right_delay))
        output = np.column_stack(
            (
                self._delay_channel(audio[:, 0], left_samples),
                self._delay_channel(audio[:, 1], right_samples),
            )
        )
        return output.astype(np.float32)
