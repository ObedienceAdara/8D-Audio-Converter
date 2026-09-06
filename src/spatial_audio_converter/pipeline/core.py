from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..analysis.signal import SignalAnalyzer
from ..analysis.waveform import summarize_waveform
from ..audio.decoder import AudioDecoder
from ..audio.encoder import AudioEncoder
from ..audio.metadata import MetadataExtractor
from ..config import AudioProcessingConfig, PipelineOptions
from ..domain.models import PipelineArtifacts
from ..effects.reverb import RoomReverb
from ..processing.loudness import LoudnessController
from ..processing.quality import QualityAnalyzer
from ..processing.report import write_quality_html, write_quality_report
from ..spatial.engine import SpatialEngine

ProgressCallback = Callable[[int, str], None]


class AudioPipeline:
    """Orchestrate decode → binaural spatialization → room → loudness → QA → encode."""

    def __init__(self, options: PipelineOptions | None = None) -> None:
        self.options = options or PipelineOptions()
        self.decoder = AudioDecoder()
        self.metadata = MetadataExtractor()
        self.analyzer = SignalAnalyzer()
        self.spatial = SpatialEngine(self.options.block_size, self.options.hrtf_ir_length)
        self.loudness = LoudnessController()
        self.quality = QualityAnalyzer()
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

        self._report(progress_callback, 12, "Analyzing source")
        analysis = self.analyzer.analyze(source) if self.options.analysis_enabled else None
        metadata = self.metadata.extract(source, input_path) if self.options.metadata_enabled else {}

        self._report(progress_callback, 22, "Binaural HRTF rendering")
        hrtf_path = config.hrtf_sofa_path if config.hrtf_source == "sofa" else None
        spatial = self.spatial.process(
            source,
            config.pan_speed_hz,
            config.depth,
            use_hrtf=config.hrtf_enabled and config.spatial_mode == "binaural",
            headphone_mode=config.headphone_mode and config.spatial_mode == "binaural",
            hrtf_path=hrtf_path,
        )

        room = RoomReverb(config.room_ir_path if config.room_model == "measured-wav" else None)
        if config.room_enabled and config.reverb_mix > 0:
            self._report(progress_callback, 55, "Applying room simulation")
            effected = room.process(
                spatial.samples,
                spatial.sample_rate,
                config.reverb_delay_ms,
                config.reverb_decay,
                config.reverb_mix,
                config.room_size,
                config.room_damping,
                config.room_model,
            )
        else:
            room.last_diagnostics = {"model": "disabled", "ir_duration_seconds": 0.0}
            effected = spatial.samples.copy()

        self._report(progress_callback, 72, "Loudness control")
        controlled = self.loudness.process(effected, config.target_rms_db, config.limiter_db)
        result_buffer = spatial.copy()
        result_buffer.samples = controlled

        self._report(progress_callback, 84, "Computing quality report")
        report = self.quality.automated_report(
            source.samples,
            controlled,
            source.sample_rate,
            context={
                "renderer": "binaural-hrtf" if config.spatial_mode == "binaural" else "equal-power+ild+itd",
                "hrtf_source": self.spatial.hrtf_source if config.spatial_mode == "binaural" else "not-used",
                "room_model": room.last_diagnostics.get("model", config.room_model),
                "room_diagnostics": room.last_diagnostics,
                "headphone_mode": config.headphone_mode,
            },
        )
        final_output = Path(output_path)
        report_path = write_quality_report(report, final_output.with_suffix(final_output.suffix + ".quality.json"))
        report_html_path = write_quality_html(report, final_output.with_suffix(final_output.suffix + ".quality.html"))
        metrics: dict = {
            "duration_seconds": round(controlled.shape[0] / source.sample_rate, 4),
            "sample_rate": int(source.sample_rate),
            "input_frames": int(source.frames),
            "output_frames": int(controlled.shape[0]),
            "clipped_samples": int(np.count_nonzero(np.abs(controlled) >= 0.99999)),
            "renderer": report["context"]["renderer"],
            "hrtf_source": report["context"]["hrtf_source"],
            "room_model": report["context"]["room_model"],
            "output_format": config.output_format,
            "quality_report": report,
        }
        if analysis:
            metrics.update({f"input_{key}": value for key, value in analysis.as_dict().items()})
        metrics.update({f"after_{key}": value for key, value in report["after"].items() if not isinstance(value, (dict, list))})
        metrics["comparison"] = report["comparison"]

        self._report(progress_callback, 93, "Encoding output")
        final_path = self.encoder.encode(
            result_buffer,
            output_path,
            config.output_format,
            bitrate=config.output_bitrate,
        )
        waveform = summarize_waveform(controlled)
        self._report(progress_callback, 100, "Complete")
        return PipelineArtifacts(
            str(final_path),
            metrics=metrics,
            metadata=metadata,
            waveform=waveform,
            quality_report_path=report_path,
            quality_report_html_path=report_html_path,
        )
