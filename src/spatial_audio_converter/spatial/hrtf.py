from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, resample_poly


InterpolationQuality = str


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
    def synthetic(cls, sample_rate: int = 48_000, ir_length: int = 256) -> HRTFDatabase:
        """Build a deterministic engineering HRIR bank, not a measured dataset."""
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
                max(0, round(-min(itd, 0.0) * sample_rate)),
                max(0, round(max(itd, 0.0) * sample_rate)),
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
                        ir[i, ear, index] += gain * 0.08 * np.exp(
                            -18.0 * (time[index] - time[direct])
                        )
        return cls(sample_rate, azimuths, np.zeros_like(azimuths), ir, "synthetic")

    @classmethod
    def from_sofa(cls, path: str | Path) -> HRTFDatabase:
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
            sample_rate = round(float(np.asarray(sofa["Data.SamplingRate"])[0]))
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

    @staticmethod
    def _normalize_azimuth(azimuth_deg: float | np.ndarray) -> np.ndarray:
        return ((np.asarray(azimuth_deg, dtype=np.float64) + 180.0) % 360.0) - 180.0

    @staticmethod
    def _unit_direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
        az = np.deg2rad(azimuth_deg)
        el = np.deg2rad(elevation_deg)
        cos_el = np.cos(el)
        return np.array([cos_el * np.cos(az), cos_el * np.sin(az), np.sin(el)], dtype=np.float64)

    def _resampled_responses(self, target_sample_rate: int) -> np.ndarray:
        if self.sample_rate == target_sample_rate:
            return np.asarray(self.ir, dtype=np.float32)
        gcd = int(np.gcd(self.sample_rate, target_sample_rate))
        up = target_sample_rate // gcd
        down = self.sample_rate // gcd
        return np.stack(
            [
                np.vstack([resample_poly(self.ir[index, ear], up, down) for ear in range(2)])
                for index in range(len(self.ir))
            ]
        ).astype(np.float32)

    def _spherical_neighbors(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        target = self._unit_direction(azimuth_deg, elevation_deg)
        points = np.stack(
            [
                self._unit_direction(az, el)
                for az, el in zip(self.azimuth_deg, self.elevation_deg, strict=True)
            ]
        )
        angular_distance = np.arccos(np.clip(points @ target, -1.0, 1.0))
        count = max(1, min(int(count), len(angular_distance)))
        indices = np.argpartition(angular_distance, count - 1)[:count]
        order = np.argsort(angular_distance[indices])
        indices = indices[order]
        return indices, angular_distance[indices]

    def _bilinear_neighbors(
        self,
        azimuth_deg: float,
        elevation_deg: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Find four grid corners when measurements form a complete az/el grid."""
        azimuth = float(self._normalize_azimuth(azimuth_deg))
        elevation = float(np.clip(elevation_deg, -90.0, 90.0))
        unique_az = np.unique(self._normalize_azimuth(self.azimuth_deg))
        unique_el = np.unique(np.asarray(self.elevation_deg, dtype=np.float64))
        if len(unique_az) < 2 or len(unique_el) < 2:
            return None

        lookup = {
            (round(float(self._normalize_azimuth(az)), 6), round(float(el), 6)): index
            for index, (az, el) in enumerate(
                zip(self.azimuth_deg, self.elevation_deg, strict=True)
            )
        }

        def bracket(values: np.ndarray, target: float) -> tuple[int, int, float]:
            if target <= values[0]:
                return 0, 0, 0.0
            if target >= values[-1]:
                last = len(values) - 1
                return last, last, 0.0
            right = int(np.searchsorted(values, target, side="right"))
            left = right - 1
            fraction = float((target - values[left]) / (values[right] - values[left]))
            return left, right, fraction

        # Expand the azimuth measurements over both sides of the circular seam.
        extended = np.sort(np.concatenate((unique_az - 360.0, unique_az, unique_az + 360.0)))
        right = int(np.searchsorted(extended, azimuth, side="right"))
        left = max(0, right - 1)
        right = min(right, len(extended) - 1)
        az0, az1 = float(extended[left]), float(extended[right])
        if az1 == az0:
            return None
        az_fraction = float((azimuth - az0) / (az1 - az0))
        el0_index, el1_index, el_fraction = bracket(unique_el, elevation)
        az0_norm = float(self._normalize_azimuth(az0))
        az1_norm = float(self._normalize_azimuth(az1))
        corners = (
            (az0_norm, unique_el[el0_index], (1.0 - az_fraction) * (1.0 - el_fraction)),
            (az1_norm, unique_el[el0_index], az_fraction * (1.0 - el_fraction)),
            (az0_norm, unique_el[el1_index], (1.0 - az_fraction) * el_fraction),
            (az1_norm, unique_el[el1_index], az_fraction * el_fraction),
        )
        indices: list[int] = []
        weights: list[float] = []
        for corner_az, corner_el, weight in corners:
            index = lookup.get((round(corner_az, 6), round(float(corner_el), 6)))
            if index is None:
                return None
            indices.append(index)
            weights.append(weight)
        return np.asarray(indices, dtype=int), np.asarray(weights, dtype=np.float64)

    def _interpolation_weights(
        self,
        azimuth_deg: float,
        elevation_deg: float,
        quality: InterpolationQuality,
        neighbors: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if quality not in {"nearest", "bilinear", "spherical"}:
            raise ValueError("quality must be 'nearest', 'bilinear', or 'spherical'")
        azimuth = float(self._normalize_azimuth(azimuth_deg))
        if quality == "nearest":
            indices, _ = self._spherical_neighbors(azimuth, elevation_deg, 1)
            return indices, np.ones(1, dtype=np.float64)
        if quality == "bilinear":
            bilinear = self._bilinear_neighbors(azimuth, elevation_deg)
            if bilinear is not None:
                indices, weights = bilinear
                total = float(np.sum(weights))
                return indices, weights / total if total > 0 else np.ones_like(weights) / len(weights)
        count = max(2, neighbors if quality == "spherical" else min(neighbors, 4))
        indices, distance = self._spherical_neighbors(azimuth, elevation_deg, count)
        if distance[0] <= 1e-9:
            return indices[:1], np.ones(1, dtype=np.float64)
        power = 2.0 if quality == "spherical" else 1.5
        weights = 1.0 / np.maximum(distance, 1e-6) ** power
        return indices, weights / np.sum(weights)

    def interpolate_ir(
        self,
        azimuth_deg: float,
        elevation_deg: float = 0.0,
        target_sample_rate: int | None = None,
        quality: InterpolationQuality = "spherical",
        neighbors: int = 4,
        fft_size: int | None = None,
    ) -> np.ndarray:
        """Interpolate an HRIR using directional weights and complex FFT spectra."""
        responses = self._resampled_responses(target_sample_rate or self.sample_rate)
        indices, weights = self._interpolation_weights(
            azimuth_deg, elevation_deg, quality, neighbors
        )
        selected = responses[indices]
        taps = max(selected.shape[-1], 1)
        nfft = int(fft_size or 2 ** int(np.ceil(np.log2(max(256, 2 * taps)))))
        spectrum = np.fft.rfft(selected, n=nfft, axis=-1)
        blended = np.sum(spectrum * weights[:, None, None], axis=0)
        interpolated = np.fft.irfft(blended, n=nfft, axis=-1)[..., :taps]
        return np.asarray(interpolated, dtype=np.float32)

    def nearest_ir(self, azimuth_deg: float, target_sample_rate: int) -> np.ndarray:
        """Backward-compatible nearest-neighbor directional lookup."""
        return self.interpolate_ir(azimuth_deg, 0.0, target_sample_rate, "nearest", 1)


class HRTFConvolver:
    """Time-varying binaural convolution with smooth directional interpolation."""

    def __init__(
        self,
        ir_length: int = 256,
        block_size: int = 1024,
        hrtf_path: str | None = None,
        interpolation_quality: InterpolationQuality = "spherical",
        interpolation_neighbors: int = 4,
        filter_crossfade_blocks: int = 2,
        trajectory_smoothing: float = 0.15,
    ) -> None:
        if interpolation_quality not in {"nearest", "bilinear", "spherical"}:
            raise ValueError("interpolation_quality must be 'nearest', 'bilinear', or 'spherical'")
        if interpolation_neighbors < 2:
            raise ValueError("interpolation_neighbors must be at least 2")
        if filter_crossfade_blocks < 0:
            raise ValueError("filter_crossfade_blocks must be non-negative")
        self.ir_length = max(64, ir_length)
        self.block_size = max(128, block_size)
        self.interpolation_quality = interpolation_quality
        self.interpolation_neighbors = int(interpolation_neighbors)
        self.filter_crossfade_blocks = int(filter_crossfade_blocks)
        self.trajectory_smoothing = float(np.clip(trajectory_smoothing, 0.0, 1.0))
        configured_path = hrtf_path or os.getenv("AUDIO_HRTF_SOFA_PATH")
        self.database = (
            HRTFDatabase.from_sofa(configured_path)
            if configured_path
            else HRTFDatabase.synthetic(ir_length=self.ir_length)
        )
        self._filter_cache: dict[tuple[int, int, str, int, int], np.ndarray] = {}

    @property
    def source(self) -> str:
        return self.database.source

    @staticmethod
    def _unwrap_degrees(values: np.ndarray) -> np.ndarray:
        return np.rad2deg(np.unwrap(np.deg2rad(values.astype(np.float64))))

    def _smooth_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        if self.trajectory_smoothing <= 0.0 or len(trajectory) < 2:
            return trajectory.astype(np.float32)
        unwrapped = self._unwrap_degrees(trajectory)
        alpha = max(0.001, 1.0 - self.trajectory_smoothing)
        smoothed = np.empty_like(unwrapped)
        smoothed[0] = unwrapped[0]
        for index in range(1, len(unwrapped)):
            smoothed[index] = smoothed[index - 1] + alpha * (
                unwrapped[index] - smoothed[index - 1]
            )
        return (((smoothed + 180.0) % 360.0) - 180.0).astype(np.float32)

    def _interpolated_filter(
        self,
        azimuth: float,
        elevation: float,
        sample_rate: int,
    ) -> np.ndarray:
        key = (
            round(float(((azimuth + 180.0) % 360.0) - 180.0) * 2.0),
            round(float(np.clip(elevation, -90.0, 90.0)) * 2.0),
            self.interpolation_quality,
            self.interpolation_neighbors,
            sample_rate,
        )
        cached = self._filter_cache.get(key)
        if cached is not None:
            return cached
        response = self.database.interpolate_ir(
            azimuth,
            elevation,
            sample_rate,
            self.interpolation_quality,
            self.interpolation_neighbors,
            fft_size=max(512, 2 * self.ir_length),
        )
        if response.shape[-1] > self.ir_length:
            response = response[:, : self.ir_length]
        peak = float(np.max(np.abs(response))) if response.size else 1.0
        if peak > 1e-9:
            response = response / peak
        result = response.astype(np.float32)
        self._filter_cache[key] = result
        if len(self._filter_cache) > 4096:
            self._filter_cache.pop(next(iter(self._filter_cache)))
        return result

    @staticmethod
    def _crossfade_filter(
        previous: np.ndarray | None,
        current: np.ndarray,
        amount: float,
    ) -> np.ndarray:
        if previous is None or amount >= 1.0:
            return current
        return ((1.0 - amount) * previous + amount * current).astype(np.float32)

    def process(
        self,
        mono: np.ndarray,
        azimuth_deg: np.ndarray,
        sample_rate: int,
        elevation_deg: np.ndarray | None = None,
    ) -> np.ndarray:
        source = np.asarray(mono, dtype=np.float32).reshape(-1)
        azimuth = np.asarray(azimuth_deg, dtype=np.float32).reshape(-1)
        elevation = (
            np.zeros_like(azimuth)
            if elevation_deg is None
            else np.asarray(elevation_deg, dtype=np.float32).reshape(-1)
        )
        if len(source) != len(azimuth) or len(source) != len(elevation):
            raise ValueError("Spatial trajectory length must match source length.")
        if source.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        smoothed_azimuth = self._smooth_trajectory(azimuth)
        previous_filter: np.ndarray | None = None
        output = np.zeros((len(source) + self.ir_length - 1, 2), dtype=np.float64)

        for block_index, start in enumerate(range(0, len(source), self.block_size)):
            stop = min(start + self.block_size, len(source))
            center = start + (stop - start) // 2
            target = self._interpolated_filter(
                float(smoothed_azimuth[center]),
                float(elevation[center]),
                sample_rate,
            )
            if previous_filter is not None and self.filter_crossfade_blocks > 0:
                amount = min(1.0, (block_index + 1) / (self.filter_crossfade_blocks + 1))
                target = self._crossfade_filter(previous_filter, target, amount)
            previous_filter = target
            for ear in range(2):
                response = fftconvolve(source[start:stop], target[ear], mode="full")
                end = min(len(output), start + len(response))
                output[start:end, ear] += response[: end - start]

        return output[: len(source)].astype(np.float32)
