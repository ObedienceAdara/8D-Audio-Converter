from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OutputFormat = Literal["mp3", "wav"]
SpatialMode = Literal["binaural", "stereo"]
RoomModel = Literal["schroeder-moorer", "measured-wav", "early-reflections"]
HRTFInterpolationQuality = Literal["nearest", "bilinear", "spherical"]


@dataclass(frozen=True, slots=True)
class AudioProcessingConfig:
    """User-controllable settings for the spatial-audio pipeline."""

    pan_speed_hz: float = 0.50
    depth: float = 0.95
    reverb_delay_ms: int = 50
    reverb_decay: float = 0.30
    reverb_mix: float = 0.30
    room_size: float = 0.70
    room_damping: float = 0.35
    room_model: RoomModel = "schroeder-moorer"
    room_ir_path: str | None = None
    room_enabled: bool = True
    target_rms_db: float = -18.0
    limiter_db: float = -1.0
    output_format: OutputFormat = "mp3"
    output_bitrate: str = "320k"
    max_duration_seconds: int = 900
    spatial_mode: SpatialMode = "binaural"
    hrtf_enabled: bool = True
    headphone_mode: bool = True
    hrtf_source: Literal["synthetic", "sofa"] = "synthetic"
    hrtf_sofa_path: str | None = None
    hrtf_interpolation_quality: HRTFInterpolationQuality = "spherical"
    hrtf_interpolation_neighbors: int = 4
    hrtf_filter_crossfade_blocks: int = 2
    hrtf_trajectory_smoothing: float = 0.15

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
        if not 0 <= self.room_size <= 1:
            raise ValueError("room_size must be in [0, 1].")
        if not 0 <= self.room_damping <= 1:
            raise ValueError("room_damping must be in [0, 1].")
        if self.room_model not in {"schroeder-moorer", "measured-wav", "early-reflections"}:
            raise ValueError("unsupported room_model")
        if self.room_model == "measured-wav" and not self.room_ir_path:
            raise ValueError("room_ir_path is required when room_model='measured-wav'.")
        if not -60 <= self.target_rms_db <= 0:
            raise ValueError("target_rms_db must be in [-60, 0].")
        if not -20 <= self.limiter_db <= 0:
            raise ValueError("limiter_db must be in [-20, 0].")
        if self.output_format not in {"mp3", "wav"}:
            raise ValueError("output_format must be 'mp3' or 'wav'.")
        if self.max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive.")
        if self.spatial_mode not in {"binaural", "stereo"}:
            raise ValueError("spatial_mode must be 'binaural' or 'stereo'.")
        if self.hrtf_source not in {"synthetic", "sofa"}:
            raise ValueError("hrtf_source must be 'synthetic' or 'sofa'.")
        if self.hrtf_source == "sofa" and not self.hrtf_sofa_path:
            raise ValueError("hrtf_sofa_path is required when hrtf_source='sofa'.")
        if self.hrtf_interpolation_quality not in {"nearest", "bilinear", "spherical"}:
            raise ValueError("unsupported hrtf_interpolation_quality")
        if not 2 <= self.hrtf_interpolation_neighbors <= 32:
            raise ValueError("hrtf_interpolation_neighbors must be in [2, 32].")
        if not 0 <= self.hrtf_filter_crossfade_blocks <= 16:
            raise ValueError("hrtf_filter_crossfade_blocks must be in [0, 16].")
        if not 0 <= self.hrtf_trajectory_smoothing <= 1:
            raise ValueError("hrtf_trajectory_smoothing must be in [0, 1].")


@dataclass(slots=True)
class PipelineOptions:
    """Runtime options that do not belong in the public processing config."""

    block_size: int = 1024
    hrtf_ir_length: int = 256
    analysis_enabled: bool = True
    metadata_enabled: bool = True
