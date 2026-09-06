from pathlib import Path
from tempfile import NamedTemporaryFile

from spatial_audio_converter.config import AudioProcessingConfig
from spatial_audio_converter.domain.models import PipelineArtifacts
from spatial_audio_converter.jobs.manager import JobManager
from spatial_audio_converter.storage.local import LocalStorage


class FakePipeline:
    def run(self, input_path, output_path, config):
        Path(output_path).write_bytes(b"processed")
        return PipelineArtifacts(str(output_path), metrics={"ok": 1}, metadata={})


def test_job_manager_runs_and_persists_job_result(tmp_path):
    storage = LocalStorage(tmp_path / "storage")
    manager = JobManager(pipeline=FakePipeline(), storage=storage, max_workers=1)
    try:
        config = AudioProcessingConfig(output_format="wav", room_enabled=False, hrtf_enabled=False)
        with NamedTemporaryFile(suffix=".wav") as source:
            source.write(b"input")
            source.flush()
            record = manager.submit(source, "input.wav", config)
            record.future.result(timeout=5)

        assert record.status == "completed"
        assert record.output_path is not None
        assert Path(record.output_path).read_bytes() == b"processed"
        assert record.metrics == {"ok": 1}
    finally:
        manager.executor.shutdown(wait=True)
