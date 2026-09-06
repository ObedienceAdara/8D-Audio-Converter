from __future__ import annotations

import numpy as np


class HRTFConvolver:
    """Blockwise binaural filter using a deterministic analytic HRTF approximation.

    The built-in profile is intentionally transparent and self-contained; it is
    not a measured HRTF dataset. Each block gets ear-specific delay, attenuation,
    and a short FIR head-shadow response derived from azimuth.
    """

    def __init__(self, ir_length: int = 96, block_size: int = 4096) -> None:
        self.ir_length = max(16, ir_length)
        self.block_size = max(256, block_size)

    def _impulse_responses(self, azimuth_deg: float, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        az = np.deg2rad(np.clip(azimuth_deg, -90.0, 90.0))
        sin_az = float(np.sin(az))
        itd_s = 0.00035 * sin_az
        base = int(round(abs(itd_s) * sample_rate))
        left_delay = base if itd_s > 0 else 0
        right_delay = base if itd_s < 0 else 0
        toward_left = (1.0 + sin_az) / 2.0
        left_gain = 0.72 + 0.28 * toward_left
        right_gain = 0.72 + 0.28 * (1.0 - toward_left)
        far_left = right_gain < left_gain
        shadow = 0.55 if far_left else 1.0

        left = np.zeros(self.ir_length, dtype=np.float32)
        right = np.zeros(self.ir_length, dtype=np.float32)
        if left_delay < self.ir_length:
            left[left_delay] = left_gain
        if right_delay < self.ir_length:
            right[right_delay] = right_gain
        # A short, decaying tail approximates the frequency shaping/head shadow.
        for k in range(1, min(12, self.ir_length)):
            tail = (0.08 / (k + 1))
            left[min(self.ir_length - 1, left_delay + k)] += tail * shadow
            right[min(self.ir_length - 1, right_delay + k)] += tail
        return left, right

    def process(self, mono: np.ndarray, azimuth_deg: np.ndarray, sample_rate: int) -> np.ndarray:
        mono = np.asarray(mono, dtype=np.float32).reshape(-1)
        azimuth_deg = np.asarray(azimuth_deg, dtype=np.float32)
        output = np.zeros((len(mono), 2), dtype=np.float32)
        overlap = np.zeros((self.ir_length - 1, 2), dtype=np.float32)
        for start in range(0, len(mono), self.block_size):
            end = min(len(mono), start + self.block_size)
            block = mono[start:end]
            az = float(azimuth_deg[min(len(azimuth_deg) - 1, start + len(block) // 2)])
            left_ir, right_ir = self._impulse_responses(az, sample_rate)
            left = np.convolve(block, left_ir, mode="full")
            right = np.convolve(block, right_ir, mode="full")
            full = np.column_stack((left, right)).astype(np.float32)
            full[: self.ir_length - 1] += overlap
            valid = min(end - start, full.shape[0])
            output[start:end] = full[:valid]
            overlap = np.zeros((self.ir_length - 1, 2), dtype=np.float32)
            tail = full[valid:]
            overlap[: min(len(overlap), len(tail))] = tail[: len(overlap)]
        return output
