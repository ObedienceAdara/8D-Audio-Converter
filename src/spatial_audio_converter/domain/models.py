from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class AudioMetadata:
    title: str = ""
    artist: str = ""
    album: str = ""
    genre: str = ""
    year: str = ""
    comment: str = ""

    def as_tags(self) -> dict[str, str]:
        tags = {}
        for key, value in {
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "genre": self.genre,
            "date": self.year,
            "comment": self.comment,
        }.items():
            if value:
                tags[key] = value
        return tags


@dataclass(slots=True)
class AudioBuffer:
    """Normalized floating-point audio with shape (frames, channels)."""

    samples: np.ndarray
    sample_rate: int
    metadata: AudioMetadata = field(default_factory=AudioMetadata)

    def __post_init__(self) -> None:
        if self.samples.ndim == 1:
            self.samples = self.samples[:, None]
        if self.samples.ndim != 2:
            raise ValueError("Audio samples must have shape (frames, channels).")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        self.samples = np.asarray(self.samples, dtype=np.float32)

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])

    @property
    def channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate

    def copy(self) -> AudioBuffer:
        return AudioBuffer(self.samples.copy(), self.sample_rate, self.metadata)


@dataclass(slots=True)
class SignalAnalysis:
    duration_seconds: float
    sample_rate: int
    channels: int
    peak: float
    rms: float
    dc_offset: float
    crest_factor_db: float
    stereo_correlation: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "duration_seconds": round(self.duration_seconds, 4),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "peak": round(self.peak, 6),
            "rms": round(self.rms, 6),
            "dc_offset": round(self.dc_offset, 6),
            "crest_factor_db": round(self.crest_factor_db, 3),
            "stereo_correlation": round(self.stereo_correlation, 6),
        }


@dataclass(slots=True)
class PipelineArtifacts:
    """Artifacts and measurements produced by a completed pipeline run."""

    output_path: str
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
