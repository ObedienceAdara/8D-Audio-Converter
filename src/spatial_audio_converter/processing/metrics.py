from __future__ import annotations

import numpy as np


class QualityMetrics:
    """Compute compact objective measurements for processed audio."""

    def compute(self, before: np.ndarray, after: np.ndarray, sample_rate: int) -> dict[str, float | int]:
        peak = float(np.max(np.abs(after))) if after.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(after)))) if after.size else 0.0
        clipping = int(np.count_nonzero(np.abs(after) >= 0.99999))
        correlation = 1.0
        if (
            after.ndim == 2
            and after.shape[1] >= 2
            and len(after) > 1
            and np.std(after[:, 0]) > 1e-12
            and np.std(after[:, 1]) > 1e-12
        ):
            correlation = float(np.corrcoef(after[:, 0], after[:, 1])[0, 1])
        return {
            "duration_seconds": round(len(after) / sample_rate, 4),
            "sample_rate": int(sample_rate),
            "peak_dbfs": round(20.0 * np.log10(max(peak, 1e-12)), 3),
            "rms_dbfs": round(20.0 * np.log10(max(rms, 1e-12)), 3),
            "stereo_correlation": round(correlation, 6),
            "clipped_samples": clipping,
            "input_frames": len(before),
            "output_frames": len(after),
        }
