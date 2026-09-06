from __future__ import annotations

import numpy as np

from .quality import QualityAnalyzer


class QualityMetrics:
    """Backward-compatible facade over the Phase-5 quality analyzer."""

    def __init__(self) -> None:
        self.analyzer = QualityAnalyzer()

    def compute(self, before: np.ndarray, after: np.ndarray, sample_rate: int) -> dict[str, object]:
        report = self.analyzer.automated_report(before, after, sample_rate)
        result = dict(report["after"])
        comparison = report["comparison"]
        result.update(
            {
                "stereo_correlation": self._stereo_correlation(after),
                "clipped_samples": int(np.count_nonzero(np.abs(after) >= 0.99999)),
                "input_frames": len(before),
                "output_frames": len(after),
                "snr_db": comparison.get("snr_db"),
                "downmix_snr_db": comparison.get("downmix_snr_db"),
                "spectral_distance_rmse_db": comparison.get("spectral_distance_rmse_db"),
                "frequency_response": result.get("frequency_response"),
            }
        )
        return result

    @staticmethod
    def _stereo_correlation(audio: np.ndarray) -> float:
        x = np.asarray(audio)
        if x.ndim != 2 or x.shape[1] < 2 or len(x) <= 1:
            return 1.0
        left = x[:, 0]
        right = x[:, 1]
        if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
            return 1.0
        return round(float(np.corrcoef(left, right)[0, 1]), 6)
