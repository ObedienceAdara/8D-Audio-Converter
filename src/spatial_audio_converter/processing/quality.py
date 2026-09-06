from __future__ import annotations

import numpy as np
import pyloudnorm as pyln
from scipy.signal import resample_poly

EPS = 1e-12


def _db(value: float, floor: float = EPS) -> float:
    return float(20.0 * np.log10(max(abs(value), floor)))


def _true_peak_dbfs(audio: np.ndarray) -> float:
    x = np.asarray(audio, dtype=np.float64)
    if not x.size:
        return -120.0
    if x.ndim == 1:
        x = x[:, None]
    peaks = []
    for channel in range(x.shape[1]):
        samples = resample_poly(x[:, channel], 4, 1)
        peaks.append(float(np.max(np.abs(samples))) if samples.size else 0.0)
    return _db(max(peaks) if peaks else 0.0)


def _spectral_profile(audio: np.ndarray, sample_rate: int, bins: int = 32) -> dict[str, list[float]]:
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    if not x.size:
        return {"frequencies_hz": [], "magnitude_db": []}
    frequencies = np.geomspace(20.0, min(20000.0, sample_rate / 2.0), bins)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) + EPS
    fft_frequencies = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
    magnitude_db = np.interp(np.log(frequencies), np.log(fft_frequencies[1:]), 20.0 * np.log10(spectrum[1:]))
    return {"frequencies_hz": [round(float(value), 3) for value in frequencies], "magnitude_db": [round(float(value), 3) for value in magnitude_db]}


def _spectral_summary(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    profile = _spectral_profile(audio, sample_rate, bins=64)
    if not profile["frequencies_hz"]:
        return {"spectral_centroid_hz": 0.0, "spectral_rolloff_hz": 0.0}
    frequencies = np.asarray(profile["frequencies_hz"])
    magnitude = 10.0 ** (np.asarray(profile["magnitude_db"]) / 20.0)
    total = float(np.sum(magnitude))
    centroid = float(np.sum(frequencies * magnitude) / max(total, EPS))
    cumulative = np.cumsum(magnitude)
    index = int(np.searchsorted(cumulative, 0.85 * cumulative[-1])) if cumulative[-1] else 0
    return {"spectral_centroid_hz": round(centroid, 3), "spectral_rolloff_hz": round(float(frequencies[index]), 3)}


def _band_energies(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    if not x.size:
        return {}
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    frequencies = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
    bands = ((20, 80), (80, 250), (250, 1000), (1000, 4000), (4000, 10000), (10000, 20000))
    result: dict[str, float] = {}
    for low, high in bands:
        mask = (frequencies >= low) & (frequencies < high)
        energy = float(np.mean(spectrum[mask])) if np.any(mask) else 0.0
        result[f"{low}_{high}_hz_db"] = round(_db(np.sqrt(max(energy, 0.0))), 3)
    return result


class QualityAnalyzer:
    """Objective loudness, dynamics, frequency-response and before/after analysis."""

    def __init__(self) -> None:
        self._meters: dict[int, pyln.Meter] = {}

    def _meter(self, sample_rate: int) -> pyln.Meter:
        if sample_rate not in self._meters:
            self._meters[sample_rate] = pyln.Meter(sample_rate, block_size=0.4, overlap=0.75)
        return self._meters[sample_rate]

    def integrated_lufs(self, audio: np.ndarray, sample_rate: int) -> float:
        x = np.asarray(audio, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if len(x) < int(0.4 * sample_rate):
            return float("nan")
        try:
            return float(self._meter(sample_rate).integrated_loudness(x))
        except ValueError:
            return float("nan")

    def analyze(self, audio: np.ndarray, sample_rate: int) -> dict[str, object]:
        x = np.asarray(audio, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        crest = 20.0 * np.log10(max(peak / max(rms, EPS), EPS))
        lufs = self.integrated_lufs(x, sample_rate)
        return {
            "duration_seconds": round(len(x) / sample_rate, 4),
            "sample_rate": int(sample_rate),
            "channels": int(x.shape[1]),
            "rms_dbfs": round(_db(rms), 3),
            "peak_dbfs": round(_db(peak), 3),
            "true_peak_dbfs": round(_true_peak_dbfs(x), 3),
            "crest_factor_db": round(float(crest), 3),
            "integrated_lufs": None if np.isnan(lufs) else round(lufs, 3),
            "spectral": _spectral_summary(x, sample_rate),
            "frequency_response": _spectral_profile(x, sample_rate),
            "bands": _band_energies(x, sample_rate),
        }

    def compare(self, before: np.ndarray, after: np.ndarray, sample_rate: int) -> dict[str, object]:
        a = np.asarray(before, dtype=np.float64)
        b = np.asarray(after, dtype=np.float64)
        if a.ndim == 1:
            a = a[:, None]
        if b.ndim == 1:
            b = b[:, None]
        n = min(len(a), len(b))
        reference = np.mean(a[:n], axis=1)
        rendered = np.mean(b[:n], axis=1)
        residual = rendered - reference
        signal_power = float(np.mean(reference**2))
        noise_power = float(np.mean(residual**2))
        snr = 10.0 * np.log10(max(signal_power, EPS) / max(noise_power, EPS))
        spectral_before = _spectral_profile(reference, sample_rate)
        spectral_after = _spectral_profile(rendered, sample_rate)
        before_mag = np.asarray(spectral_before["magnitude_db"])
        after_mag = np.asarray(spectral_after["magnitude_db"])
        spectral_distance = float(np.sqrt(np.mean(np.square(after_mag - before_mag)))) if len(before_mag) else 0.0
        band_before = _band_energies(reference, sample_rate)
        band_after = _band_energies(rendered, sample_rate)
        spectral_delta = {
            key: round(band_after[key] - band_before[key], 3)
            for key in band_before
            if key in band_after
        }
        return {
            "duration_delta_seconds": round(len(b) / sample_rate - len(a) / sample_rate, 4),
            "downmix_snr_db": round(float(snr), 3),
            "spectral_distance_rmse_db": round(spectral_distance, 3),
            "spectral_centroid_delta_hz": round(
                _spectral_summary(rendered, sample_rate)["spectral_centroid_hz"]
                - _spectral_summary(reference, sample_rate)["spectral_centroid_hz"],
                3,
            ),
            "spectral_rolloff_delta_hz": round(
                _spectral_summary(rendered, sample_rate)["spectral_rolloff_hz"]
                - _spectral_summary(reference, sample_rate)["spectral_rolloff_hz"],
                3,
            ),
            "band_level_delta_db": spectral_delta,
        }

    def automated_report(self, before: np.ndarray, after: np.ndarray, sample_rate: int, context: dict | None = None) -> dict:
        return {
            "report_version": 1,
            "standard_reference": "ITU-R BS.1770-style integrated loudness via pyloudnorm",
            "context": context or {},
            "before": self.analyze(before, sample_rate),
            "after": self.analyze(after, sample_rate),
            "comparison": self.compare(before, after, sample_rate),
        }
