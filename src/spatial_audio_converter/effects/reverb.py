from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, lfilter, resample_poly


class RoomReverb:
    """Early reflections plus Schroeder/Moorer-style late reverberation."""

    def __init__(self, room_ir_path: str | None = None) -> None:
        self.room_ir_path = room_ir_path
        self.room_model = "schroeder-moorer"
        self.last_diagnostics: dict[str, float | str] = {}

    @staticmethod
    def _load_measured_ir(path: str | Path, sample_rate: int) -> np.ndarray:
        source_rate, data = wavfile.read(path)
        x = np.asarray(data, dtype=np.float32)
        if np.issubdtype(data.dtype, np.integer):
            x /= max(float(np.iinfo(data.dtype).max), 1.0)
        if x.ndim == 1:
            x = x[:, None]
        if source_rate != sample_rate:
            gcd = math.gcd(int(source_rate), int(sample_rate))
            up = sample_rate // gcd
            down = source_rate // gcd
            x = np.column_stack([resample_poly(x[:, ch], up, down) for ch in range(x.shape[1])])
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        peak = float(np.max(np.abs(x))) if x.size else 1.0
        return (x / peak).astype(np.float32) if peak > 0 else x

    def _early_ir(self, sample_rate: int, delay_ms: int, decay: float, room_size: float) -> np.ndarray:
        base = max(1, int(sample_rate * delay_ms / 1000))
        scale = 0.65 + 1.35 * room_size
        reflections = (
            (base, 0.50),
            (max(1, int(base * 1.37 * scale)), 0.38),
            (max(1, int(base * 1.91 * scale)), 0.28),
            (max(1, int(base * 2.53 * scale)), 0.20),
        )
        length = max(index for index, _ in reflections) + 1
        ir = np.zeros((length, 2), dtype=np.float32)
        ir[0] = 1.0
        for index, gain in reflections:
            ir[index, 0] += decay * gain * 0.92
            ir[index, 1] += decay * gain * 1.08
        return ir

    @staticmethod
    def _comb_ir(length: int, delay: int, gain: float) -> np.ndarray:
        impulse = np.zeros(length, dtype=np.float32)
        impulse[0] = 1.0
        denominator = [1.0] + [0.0] * (delay - 1) + [-gain]
        return lfilter([1.0], denominator, impulse).astype(np.float32)

    @staticmethod
    def _allpass(signal: np.ndarray, delay: int, gain: float) -> np.ndarray:
        output = np.zeros_like(signal)
        for index in range(len(signal)):
            delayed_input = signal[index - delay] if index >= delay else 0.0
            delayed_output = output[index - delay] if index >= delay else 0.0
            output[index] = -gain * signal[index] + delayed_input + gain * delayed_output
        return output

    def _algorithmic_ir(self, sample_rate: int, delay_ms: int, decay: float, room_size: float, damping: float) -> np.ndarray:
        early = self._early_ir(sample_rate, delay_ms, decay, room_size)
        if self.room_model == "early-reflections":
            return early
        scale = 0.65 + 1.35 * room_size
        delays_ms = (29.7, 37.1, 41.1, 43.7)
        delays = [max(1, int(sample_rate * ms * scale / 1000)) for ms in delays_ms]
        gains = [0.78, 0.73, 0.69, 0.65]
        tail_seconds = 1.2 + 1.8 * room_size
        length = max(len(early), int(sample_rate * tail_seconds))
        late = np.zeros(length, dtype=np.float32)
        effective_decay = max(0.05, decay)
        for delay, gain in zip(delays, gains):
            late += self._comb_ir(length, delay, gain * effective_decay)
        late /= len(delays)
        damping_alpha = min(0.995, max(0.05, 0.50 + 0.45 * damping))
        late = lfilter([1.0 - damping_alpha], [1.0, -damping_alpha], late).astype(np.float32)
        late = self._allpass(late, max(1, int(sample_rate * 5.0 / 1000)), min(0.7, 0.35 + 0.25 * effective_decay))
        ir = np.zeros((length, 2), dtype=np.float32)
        ir[: len(early)] += early
        ir[:, 0] += 0.70 * late
        ir[:, 1] += 0.82 * late
        return ir

    @staticmethod
    def _block_convolve(signal: np.ndarray, impulse_response: np.ndarray, block_size: int = 8192) -> np.ndarray:
        output = np.zeros(len(signal), dtype=np.float32)
        for start in range(0, len(signal), block_size):
            stop = min(start + block_size, len(signal))
            response = fftconvolve(signal[start:stop], impulse_response, mode="full")
            end = min(len(output), start + len(response))
            output[start:end] += response[: end - start]
        return output

    def process(
        self,
        stereo: np.ndarray,
        sample_rate: int,
        delay_ms: int,
        decay: float,
        mix: float,
        room_size: float = 0.7,
        damping: float = 0.35,
        room_model: str = "schroeder-moorer",
    ) -> np.ndarray:
        dry = np.asarray(stereo, dtype=np.float32)
        if dry.ndim != 2 or dry.shape[1] != 2:
            raise ValueError("RoomReverb expects stereo audio with shape (frames, 2).")
        self.room_model = room_model
        if mix <= 0 or decay <= 0:
            self.last_diagnostics = {"model": "disabled", "ir_duration_seconds": 0.0}
            return dry.copy()
        if room_model == "measured-wav":
            if not self.room_ir_path:
                raise ValueError("room_ir_path is required for measured-wav room model")
            ir = self._load_measured_ir(self.room_ir_path, sample_rate)
            model = "measured-wav-ir"
        else:
            ir = self._algorithmic_ir(sample_rate, delay_ms, decay, room_size, damping)
            model = room_model
        peak = float(np.max(np.abs(ir))) if ir.size else 1.0
        if peak > 1e-9:
            ir = ir / peak
        wet = np.column_stack(
            [self._block_convolve(dry[:, channel], ir[:, channel]) for channel in range(2)]
        )
        self.last_diagnostics = {
            "model": model,
            "ir_length_samples": int(len(ir)),
            "ir_duration_seconds": float(len(ir) / sample_rate),
        }
        return ((1.0 - mix) * dry + mix * wet).astype(np.float32)
