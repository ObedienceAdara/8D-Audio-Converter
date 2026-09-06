from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, lfilter, resample_poly


class RoomReverb:
    """Early reflections plus Schroeder/Moorer-style late reverberation.

    A measured mono/stereo WAV impulse response can be supplied to replace the
    algorithmic room model. The public mix remains dry/wet and bounded.
    """

    def __init__(self, room_ir_path: str | None = None) -> None:
        self.room_ir_path = room_ir_path
        self.last_diagnostics: dict[str, float | str] = {}

    @staticmethod
    def _load_measured_ir(path: str | Path, sample_rate: int) -> np.ndarray:
        source_rate, data = wavfile.read(path)
        x = np.asarray(data, dtype=np.float32)
        if np.issubdtype(data.dtype, np.integer):
            scale = float(np.iinfo(data.dtype).max)
            x /= max(scale, 1.0)
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
        if peak > 0:
            x /= peak
        return x.astype(np.float32)

    def _algorithmic_ir(
        self,
        sample_rate: int,
        delay_ms: int,
        decay: float,
        room_size: float,
        damping: float,
    ) -> np.ndarray:
        base = max(1, int(sample_rate * delay_ms / 1000))
        scale = 0.6 + 1.4 * room_size
        early_delays = [base, int(base * 1.37 * scale), int(base * 1.91 * scale), int(base * 2.53 * scale)]
        early_gains = [0.50 * decay, 0.38 * decay, 0.28 * decay, 0.20 * decay]
        early_length = early_delays[-1] + 1
        ir = np.zeros((early_length, 2), dtype=np.float32)
        ir[0] = 1.0
        for index, gain in zip(early_delays, early_gains):
            index = min(index, early_length - 1)
            ir[index, 0] += gain * 0.92
            ir[index, 1] += gain * 1.08

        comb_delays = [29, 37, 41, 43]
        comb_delays = [max(1, int(sample_rate * d / 1000 * scale / 40.0)) for d in comb_delays]
        comb_gains = [0.74 * decay, 0.70 * decay, 0.67 * decay, 0.63 * decay]
        allpass_delay = max(1, int(sample_rate * 5.0 / 1000))
        allpass_gain = min(0.7, 0.35 + 0.25 * decay)
        length = max(early_length, max(comb_delays) + int(sample_rate * 1.2 * (0.4 + room_size)))
        input_ir = np.zeros(length, dtype=np.float32)
        input_ir[:early_length] = np.mean(ir, axis=1)
        late = np.zeros(length, dtype=np.float32)
        impulse = np.zeros(length, dtype=np.float32)
        impulse[0] = 1.0
        for delay, gain in zip(comb_delays, comb_gains):
            comb_ir = np.zeros(length, dtype=np.float32)
            comb_ir[::delay] = gain ** np.arange(len(comb_ir[::delay]), dtype=np.float32)
            late += lfilter([1.0], [1.0] + [-0.0] * (delay - 1) + [-gain], impulse)
            late = np.maximum(late, 0.0) + fftconvolve(impulse, comb_ir, mode="same") * 0.15
        alpha = min(0.95, max(0.05, 1.0 - damping * 0.85))
        filtered = lfilter([1.0 - alpha], [1.0, -alpha], late)
        ap = filtered.copy()
        for n in range(allpass_delay, len(ap)):
            ap[n] = -allpass_gain * ap[n - allpass_delay] + allpass_gain * filtered[n] + filtered[n - allpass_delay]
        left = input_ir + 0.65 * ap
        right = input_ir + 0.80 * ap
        return np.column_stack((left, right)).astype(np.float32)

    def process(
        self,
        stereo: np.ndarray,
        sample_rate: int,
        delay_ms: int,
        decay: float,
        mix: float,
        room_size: float = 0.7,
        damping: float = 0.35,
    ) -> np.ndarray:
        dry = np.asarray(stereo, dtype=np.float32)
        if dry.ndim != 2 or dry.shape[1] != 2:
            raise ValueError("RoomReverb expects stereo audio with shape (frames, 2).")
        if mix <= 0 or decay <= 0:
            self.last_diagnostics = {"model": "disabled", "rt60_proxy_seconds": 0.0}
            return dry.copy()

        if self.room_ir_path:
            ir = self._load_measured_ir(self.room_ir_path, sample_rate)
            model = "measured-wav-ir"
        else:
            ir = self._algorithmic_ir(sample_rate, delay_ms, decay, room_size, damping)
            model = "schroeder-moorer"
        peak = float(np.max(np.abs(ir))) if ir.size else 1.0
        if peak > 1e-9:
            ir = ir / peak

        wet = np.zeros((len(dry) + len(ir) - 1, 2), dtype=np.float32)
        for out_channel in range(2):
            ir_channel = ir[:, out_channel]
            wet[:, out_channel] += fftconvolve(dry[:, out_channel], ir_channel, mode="full")
        wet = wet[: len(dry)]
        output = ((1.0 - mix) * dry + mix * wet).astype(np.float32)
        self.last_diagnostics = {
            "model": model,
            "ir_length_samples": int(len(ir)),
            "ir_duration_seconds": float(len(ir) / sample_rate),
            "rt60_proxy_seconds": float(len(ir) / sample_rate * 0.65),
        }
        return output
