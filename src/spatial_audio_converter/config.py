from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

OutputFormat = Literal["mp3", "wav"]


@dataclass(frozen=True, slots=True)
class AudioProcessingConfig:
    """User-controllable settings for the spatial-audio pipeline."""

    pan_speed_hz: float = 0.50
    depth: float = 0.95
    reverb_delay_ms: int = 50
    reverb_decay: float = 0.30
    reverb_mix: float = 0.30
    target_rms_db: float = -18.0
    limiter_db: float = -1.0
    output_format: OutputFormat = "mp3"
    output_bitrate: str = "320k"
    max_duration_seconds: int = 900
    room_enabled: bool = True
    hrtf_enabled: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.pan_speed_hz <= 2.0:
            raise ValueError("pan_speed_hz must be in (0, 2].")
        if not 0 <= self.depth <= 1:
            raise ValueError("depth must be in [0, 1].")
        if not 1 <= self.reverb_delay_ms <= 500:
            raise ValueError("reverb_delay_ms must be in [1, 500].")
        if not 0 <= self.reverb_decay <= 1:
            raise ValueError("reverb_decay must be in [0, 1].")
        if not 0 <= self.reverb_mix <= 1:
            raise ValueError("reverb_mix must be in [0, 1].")
        if not -60 <= self.target_rms_db <= 0:
            raise ValueError("target_rms_db must be in [-60, 0].")
        if not -20 <= self.limiter_db <= 0:
            raise ValueError("limiter_db must be in [-20, 0].")
        if self.output_format not in {"mp3", "wav"}:
            raise ValueError("output_format must be 'mp3' or 'wav'.")
        if self.max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive.")


@dataclass(slots=True)
class PipelineOptions:
    """Runtime options that do not belong in the public processing config."""

    block_size: int = 4096
    hrtf_ir_length: int = 96
    analysis_enabled: bool = True
    metadata_enabled: bool = True


@dataclass(slots=True)
class PipelineArtifacts:
    """Artifacts and measurements produced by a completed pipeline run."""

    output_path: str
    metrics: dict[str, float | int | str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
