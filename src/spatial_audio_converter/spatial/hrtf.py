from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, resample_poly


@dataclass(frozen=True, slots=True)
class HRTFDatabase:
    """Binaural FIR database with azimuth/elevation metadata."""

    sample_rate: int
    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray
    ir: np.ndarray
    source: str

    def __post_init__(self) -> None:
        if self.ir.ndim != 3 or self.ir.shape[1] != 2:
            raise ValueError("HRTF IR must have shape (measurements, 2 ears, taps).")
        if self.ir.shape[0] != len(self.azimuth_deg) or self.ir.shape[0] != len(self.elevation_deg):
            raise ValueError("HRTF direction metadata does not match IR measurements.")
        if self.sample_rate <= 0:
            raise ValueError("HRTF sample rate must be positive.")

    @classmethod
    def synthetic(cls, sample_rate: int = 48_000, ir_length: int = 256) -> "HRTFDatabase":
        """Build a deterministic convolutional headphone-oriented HRIR bank.

        This is an engineering fallback, not a measured human HRTF dataset.
        """
        if ir_length < 64:
            raise ValueError("ir_length must be at least 64 samples")
        azimuths = np.arange(-90.0, 91.0, 15.0, dtype=np.float64)
        head_radius = 0.0875
        sound_speed = 343.0
        max_itd = head_radius / sound_speed
        ir = np.zeros((len(azimuths), 2, ir_length), dtype=np.float32)
        time = np.arange(ir_length, dtype=np.float64) / sample_rate
        for i, azimuth in enumerate(azimuths):
            lateral = float(np.sin(np.deg2rad(azimuth)))
            itd = max_itd * lateral
            delays = (
                max(0, int(round(-min(itd, 0.0) * sample_rate))),
                max(0, int(round(max(itd, 0.0) * sample_rate))),
            )
            shadow = 10.0 ** (-(1.5 + 9.0 * abs(lateral)) / 20.0)
            for ear, delay, near in ((0, delays[0], azimuth <= 0), (1, delays[1], azimuth >= 0)):
                gain = 1.0 if near else shadow
                direct = min(delay + 8, ir_length - 1)
                ir[i, ear, direct] += gain
                for offset, echo_gain in ((24, 0.12), (47, 0.07), (79, 0.045)):
                    index = min(direct + offset, ir_length - 1)
                    sign = -1.0 if ((ear + offset // 24) % 2) else 1.0
                    ir[i, ear, index] += gain * echo_gain * sign
                if not near:
                    for index in range(direct + 1, min(direct + 9, ir_length)):
                        ir[i, ear, index] += gain * 0.08 * np.exp(-18.0 * (time[index] - time[direct]))
        return cls(sample_rate, azimuths, np.zeros_like(azimuths), ir, "synthetic")

    @classmethod
    def from_sofa(cls, path: str | Path) -> "HRTFDatabase":
        """Load a measured SOFA SimpleFreeFieldHRIR/FreeFieldHRIR file."""
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("h5py is required to load SOFA HRTF files.") from exc
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"SOFA HRTF file does not exist: {file_path}")
        with h5py.File(file_path, "r") as sofa:
            required = {"Data.IR", "SourcePosition", "Data.SamplingRate"}
            missing = sorted(required - set(sofa.keys()))
            if missing:
                raise ValueError(f"SOFA file is missing datasets: {missing}")
            ir = np.asarray(sofa["Data.IR"], dtype=np.float32)
            positions = np.asarray(sofa["SourcePosition"], dtype=np.float64)
            sample_rate = int(round(float(np.asarray(sofa["Data.SamplingRate"])[0])))
            convention = sofa.attrs.get("SOFAConventions", b"")
            if isinstance(convention, bytes):
                convention = convention.decode("utf-8", errors="ignore")
        if ir.ndim != 3 or ir.shape[1] != 2:
            raise ValueError("Only two-receiver binaural SOFA IRs are supported.")
        if positions.ndim != 2 or positions.shape[1] < 2 or len(positions) != ir.shape[0]:
            raise ValueError("SOFA SourcePosition does not match Data.IR measurements.")
        if convention and "FreeFieldHRIR" not in str(convention):
            raise ValueError(f"Unsupported SOFA convention: {convention}")
        return cls(sample_rate, positions[:, 0], positions[:, 1], ir, f"sofa:{file_path.name}")

    def nearest_ir(self, azimuth_deg: float, target_sample_rate: int) -> np.ndarray:
        azimuth = np.asarray(self.azimuth_deg, dtype=np.float64)
        elevation = np.asarray(self.elevation_deg, dtype=np.float64)
        circular = np.rad2deg(np.angle(np.exp(1j * np.deg2rad(azimuth - float(azimuth_deg)))))
        index = int(np.argmin(np.hypot(circular, elevation)))
        response = np.asarray(self.ir[index], dtype=np.float32)
        if self.sample_rate == target_sample_rate:
            return response
        gcd = int(np.gcd(self.sample_rate, target_sample_rate))
        up = target_sample_rate // gcd
        down = self.sample_rate // gcd
        return np.vstack([resample_poly(response[ear], up, down) for ear in range(2)]).astype(np.float32)


class HRTFConvolver:
    """Time-varying binaural convolution for headphone playback."""

    def __init__(self, ir_length: int = 256, block_size: int = 1024, hrtf_path: str | None = None) -> None:
        self.ir_length = max(64, ir_length)
        self.block_size = max(128, block_size)
        configured_path = hrtf_path or os.getenv("AUDIO_HRTF_SOFA_PATH")
        self.database = HRTFDatabase.from_sofa(configured_path) if configured_path else HRTFDatabase.synthetic(ir_length=self.ir_length)

    @property
    def source(self) -> str:
        return self.database.source

    def process(self, mono: np.ndarray, azimuth_deg: np.ndarray, sample_rate: int) -> np.ndarray:
        source = np.asarray(mono, dtype=np.float32).reshape(-1)
        azimuth = np.asarray(azimuth_deg, dtype=np.float32).reshape(-1)
        if len(source) != len(azimuth):
            raise ValueError("Azimuth trajectory length must match source length.")
        if source.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        hrir_blocks = []
        max_taps = 0
        for start in range(0, len(source), self.block_size):
            stop = min(start + self.block_size, len(source))
            hrir = self.database.nearest_ir(float(np.mean(azimuth[start:stop])), sample_rate)
            hrir_blocks.append((start, stop, hrir))
            max_taps = max(max_taps, hrir.shape[1])

        output = np.zeros((len(source) + max_taps - 1, 2), dtype=np.float64)
        for start, stop, hrir in hrir_blocks:
            for ear in range(2):
                response = fftconvolve(source[start:stop], hrir[ear], mode="full")
                output[start : start + len(response), ear] += response
        return output[: len(source)].astype(np.float32)
