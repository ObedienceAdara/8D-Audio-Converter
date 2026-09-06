from __future__ import annotations

import numpy as np


def summarize_waveform(samples: np.ndarray, points: int = 512) -> list[float]:
    """Return a compact peak envelope suitable for a JSON API and canvas rendering."""
    if points <= 0:
        raise ValueError("points must be positive")
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 2:
        data = np.max(np.abs(data), axis=1)
    elif data.ndim == 1:
        data = np.abs(data)
    else:
        raise ValueError("samples must be one- or two-dimensional")
    if data.size == 0:
        return []

    edges = np.linspace(0, data.size, min(points, data.size) + 1, dtype=int)
    envelope: list[float] = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            continue
        envelope.append(float(np.max(data[start:end])))
    return envelope
