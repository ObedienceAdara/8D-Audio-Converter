from __future__ import annotations

from pathlib import Path

from ..config import AudioProcessingConfig, PipelineOptions
from ..analysis.signal import SignalAnalyzer
from ..audio.decoder import AudioDecoder
from ..audio.encoder import AudioEncoder
from ..audio.metadata import MetadataExtractor
from ..domain.models import PipelineArtifacts
from ..effects.reverb import RoomReverb
from ..processing.loudness import LoudnessController
from ..processing.metrics import QualityMetrics
from ..spatial.engine import SpatialEngine


class AudioPipeline:
    """Orchestrate decode → analysis → spatial → room → loudness → metrics → encode."""

    def __init__(self, options: PipelineOptions | None = None) -> None:
        self.options = options or PipelineOptions()
        self.decoder = AudioDecoder()
        self.metadata = MetadataExtractor()
        self.analyzer = SignalAnalyzer()
        self.spatial = SpatialEngine(self.options.block_size, self.options.hrtf_ir_length)
        self.room = RoomReverb()
        self.loudness = LoudnessController()
        self.metrics = QualityMetrics()
        self.encoder = AudioEncoder()

    def run(self, input_path: str | Path, output_path: str | Path, config: AudioProcessingConfig) -> PipelineArtifacts:
        source = self.decoder.decode(input_path, config.max_duration_seconds)
        analysis = self.analyzer.analyze(source) if self.options.analysis_enabled else None
        metadata = self.metadata.extract(source, input_path) if self.options.metadata_enabled else {}

        spatial = self.spatial.process(
            source,
            config.pan_speed_hz,
            config.depth,
            use_hrtf=config.hrtf_enabled,
        )
        effected = (
            self.room.process(
                spatial.samples,
                spatial.sample_rate,
                config.reverb_delay_ms,
                config.reverb_decay,
                config.reverb_mix,
            )
            if config.room_enabled
            else spatial.samples
        )
        controlled = self.loudness.process(effected, config.target_rms_db, config.limiter_db)
        result_buffer = spatial.copy()
        result_buffer.samples = controlled

        metrics = self.metrics.compute(source.samples[:, :2] if source.channels >= 2 else source.samples, controlled, source.sample_rate)
        if analysis:
            metrics.update({f"input_{k}": v for k, v in analysis.as_dict().items()})

        final_path = self.encoder.encode(
            result_buffer,
            output_path,
            config.output_format,
            bitrate=config.output_bitrate,
        )
        return PipelineArtifacts(str(final_path), metrics=metrics, metadata=metadata)
