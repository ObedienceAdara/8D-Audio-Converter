from __future__ import annotations

import numpy as np

from ..domain.models import AudioBuffer, SignalAnalysis


class SignalAnalyzer:
    """Compute inexpensive, deterministic signal diagnostics."""

    def analyze(self, audio: AudioBuffer) -> SignalAnalysis:
        x = audio.samples
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
        dc = float(np.mean(x)) if x.size else 0.0
        crest = 20.0 * np.log10(max(peak, 1e-12) / max(rms, 1e-12))
        if audio.channels >= 2 and audio.frames > 1:
            left = x[:, 0]
            right = x[:, 1]
            if np.std(left) > 1e-12 and np.std(right) > 1e-12:
                corr = float(np.corrcoef(left, right)[0, 1])
            else:
                corr = 1.0
        else:
            corr = 1.0
        return SignalAnalysis(
            duration_seconds=audio.duration_seconds,
            sample_rate=audio.sample_rate,
            channels=audio.channels,
            peak=peak,
            rms=rms,
            dc_offset=dc,
            crest_factor_db=crest,
            stereo_correlation=corr,
        )
