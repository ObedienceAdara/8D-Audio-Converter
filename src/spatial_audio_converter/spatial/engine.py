from __future__ import annotations

import numpy as np

from ..domain.models import AudioBuffer
from .hrtf import HRTFConvolver
from .ild import InterauralLevelDifference
from .itd import InterauralTimeDifference
from .panning import EqualPowerPanner
from .trajectory import TrajectoryGenerator


class SpatialEngine:
    """Compose spatial motion, legacy stereo mode, and binaural HRTF rendering."""

    def __init__(self, block_size: int = 1024, hrtf_ir_length: int = 256, hrtf_path: str | None = None) -> None:
        self.trajectory = TrajectoryGenerator()
        self.panner = EqualPowerPanner()
        self.ild = InterauralLevelDifference()
        self.itd = InterauralTimeDifference()
        self._block_size = block_size
        self._hrtf_ir_length = hrtf_ir_length
        self._hrtf_cache: dict[tuple[str | None, str, int, int, float], HRTFConvolver] = {}
        self._get_hrtf(hrtf_path, "spherical", 4, 2, 0.15)

    def _get_hrtf(
        self,
        hrtf_path: str | None,
        interpolation_quality: str,
        interpolation_neighbors: int,
        filter_crossfade_blocks: int,
        trajectory_smoothing: float,
    ) -> HRTFConvolver:
        key = (
            hrtf_path,
            interpolation_quality,
            int(interpolation_neighbors),
            int(filter_crossfade_blocks),
            round(float(trajectory_smoothing), 6),
        )
        if key not in self._hrtf_cache:
            self._hrtf_cache[key] = HRTFConvolver(
                ir_length=self._hrtf_ir_length,
                block_size=self._block_size,
                hrtf_path=hrtf_path,
                interpolation_quality=interpolation_quality,
                interpolation_neighbors=interpolation_neighbors,
                filter_crossfade_blocks=filter_crossfade_blocks,
                trajectory_smoothing=trajectory_smoothing,
            )
        return self._hrtf_cache[key]

    @property
    def hrtf_source(self) -> str:
        return next(iter(self._hrtf_cache.values())).source

    def process(
        self,
        audio: AudioBuffer,
        speed_hz: float,
        depth: float,
        use_hrtf: bool = False,
        headphone_mode: bool = False,
        hrtf_path: str | None = None,
        interpolation_quality: str = "spherical",
        interpolation_neighbors: int = 4,
        filter_crossfade_blocks: int = 2,
        trajectory_smoothing: float = 0.15,
    ) -> AudioBuffer:
        mono = audio.samples[:, 0] if audio.channels == 1 else np.mean(audio.samples[:, :2], axis=1)
        azimuth = self.trajectory.generate(audio.frames, audio.sample_rate, speed_hz, depth)
        if use_hrtf or headphone_mode:
            hrtf = self._get_hrtf(
                hrtf_path,
                interpolation_quality,
                interpolation_neighbors,
                filter_crossfade_blocks,
                trajectory_smoothing,
            )
            output = hrtf.process(mono, azimuth, audio.sample_rate)
        else:
            stereo = self.panner.pan(mono, azimuth)
            left_gain, right_gain = self.ild.gains(azimuth)
            stereo[:, 0] *= left_gain
            stereo[:, 1] *= right_gain
            output = self.itd.apply(stereo, azimuth, audio.sample_rate)
        return AudioBuffer(output, audio.sample_rate, audio.metadata)
