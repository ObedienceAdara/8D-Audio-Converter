from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..analysis.signal import SignalAnalyzer
from ..analysis.waveform import summarize_waveform
from ..audio.decoder import AudioDecoder
from ..audio.encoder import AudioEncoder
from ..audio.metadata import MetadataExtractor
from ..config import AudioProcessingConfig, PipelineOptions
from ..domain.models import PipelineArtifacts
from ..effects.reverb import RoomReverb
from ..processing.loudness import LoudnessController
from ..processing.metrics import QualityMetrics
from ..spatial.engine import SpatialEngine

ProgressCallback = Callable[[int, str], None]


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

    @staticmethod
    def _report(callback: ProgressCallback | None, progress: int, stage: str) -> None:
        if callback:
            callback(max(0, min(progress, 100)), stage)

    def run(
        self,
        input_path: str | Path,
        output_path: str | Path,
        config: AudioProcessingConfig,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineArtifacts:
        self._report(progress_callback, 2, "Decoding input")
        source = self.decoder.decode(input_path, config.max_duration_seconds)

        self._report(progress_callback, 15, "Analyzing source")
        analysis = self.analyzer.analyze(source) if self.options.analysis_enabled else None
        metadata = self.metadata.extract(source, input_path) if self.options.metadata_enabled else {}

        self._report(progress_callback, 22, "Generating spatial trajectory")
        spatial = self.spatial.process(
            source,
            config.pan_speed_hz,
            config.depth,
            use_hrtf=config.hrtf_enabled,
        )

        self._report(progress_callback, 60, "Applying spatial room model")
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

        self._report(progress_callback, 75, "Normalizing output")
        controlled = self.loudness.process(effected, config.target_rms_db, config.limiter_db)
        result_buffer = spatial.copy()
        result_buffer.samples = controlled

        self._report(progress_callback, 86, "Computing quality metrics")
        reference = source.samples[:, :2] if source.channels >= 2 else source.samples
        metrics = self.metrics.compute(reference, controlled, source.sample_rate)
        if analysis:
            metrics.update({f"input_{key}": value for key, value in analysis.as_dict().items()})
        metrics["renderer"] = "analytic-hrtf" if config.hrtf_enabled else "equal-power+ild+itd"
        metrics["output_format"] = config.output_format

        self._report(progress_callback, 92, "Encoding output")
        final_path = self.encoder.encode(
            result_buffer,
            output_path,
            config.output_format,
            bitrate=config.output_bitrate,
        )
        waveform = summarize_waveform(controlled)
        self._report(progress_callback, 100, "Complete")
        return PipelineArtifacts(str(final_path), metrics=metrics, metadata=metadata, waveform=waveform)
