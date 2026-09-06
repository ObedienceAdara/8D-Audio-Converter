import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Event

import pytest

from spatial_audio_converter.config import AudioProcessingConfig
from spatial_audio_converter.domain.models import PipelineArtifacts
from spatial_audio_converter.jobs.manager import JobManager
from spatial_audio_converter.jobs.queue import InProcessJobQueue, QueueFullError
from spatial_audio_converter.storage.local import LocalStorage


class FakePipeline:
    def run(self, input_path, output_path, config, progress_callback=None):
        for progress, stage in [(20, "fake decode"), (60, "fake DSP"), (95, "fake encode")]:
            if progress_callback:
                progress_callback(progress, stage)
        Path(output_path).write_bytes(b"processed")
        if progress_callback:
            progress_callback(100, "Complete")
        return PipelineArtifacts(str(output_path), metrics={"ok": 1}, metadata={}, waveform=[0.1, 0.2])


class BlockingPipeline:
    def __init__(self):
        self.started = Event()
        self.release = Event()

    def run(self, input_path, output_path, config, progress_callback=None):
        self.started.set()
        self.release.wait(timeout=5)
        Path(output_path).write_bytes(b"processed")
        return PipelineArtifacts(str(output_path))


def test_bounded_queue_rejects_overflow():
    queue = InProcessJobQueue(maxsize=1)
    queue.put(object())
    with pytest.raises(QueueFullError):
        queue.put(object())


def test_job_manager_requires_positive_worker_count(tmp_path):
    with pytest.raises(ValueError, match="max_workers"):
        JobManager(storage=LocalStorage(tmp_path / "storage"), max_workers=0)


def test_job_manager_submit_returns_before_conversion_finishes(tmp_path):
    pipeline = BlockingPipeline()
    manager = JobManager(
        pipeline=pipeline,
        storage=LocalStorage(tmp_path / "storage"),
        max_workers=1,
        queue_size=2,
    )
    try:
        config = AudioProcessingConfig(output_format="wav", room_enabled=False, hrtf_enabled=False)
        with NamedTemporaryFile(suffix=".wav") as source:
            source.write(b"input")
            source.flush()
            started = time.monotonic()
            record = manager.submit(source, "input.wav", config)
            submission_time = time.monotonic() - started
            assert submission_time < 1.0
            assert pipeline.started.wait(timeout=2)
            assert record.status in {"queued", "processing"}
            assert not record.done_event.is_set()
            pipeline.release.set()
            manager.wait(record.id, timeout=5)
            assert record.status == "completed"
    finally:
        pipeline.release.set()
        manager.shutdown()


def test_job_manager_uses_queue_workers_and_exposes_progress(tmp_path):
    storage = LocalStorage(tmp_path / "storage")
    manager = JobManager(pipeline=FakePipeline(), storage=storage, max_workers=1, queue_size=4)
    try:
        config = AudioProcessingConfig(output_format="wav", room_enabled=False, hrtf_enabled=False)
        with NamedTemporaryFile(suffix=".wav") as source:
            source.write(b"input")
            source.flush()
            record = manager.submit(source, "input.wav", config)
            completed = manager.wait(record.id, timeout=5)

        assert completed is record
        assert record.status == "completed"
        assert record.progress == 100
        assert record.stage == "Complete"
        assert record.output_path is not None
        assert Path(record.output_path).read_bytes() == b"processed"
        assert record.metrics == {"ok": 1}
        assert record.waveform == [0.1, 0.2]
    finally:
        manager.shutdown()


def test_batch_status_aggregates_multiple_jobs(tmp_path):
    storage = LocalStorage(tmp_path / "storage")
    manager = JobManager(pipeline=FakePipeline(), storage=storage, max_workers=2, queue_size=4)
    try:
        config = AudioProcessingConfig(output_format="wav", room_enabled=False, hrtf_enabled=False)
        with NamedTemporaryFile(suffix=".wav") as first, NamedTemporaryFile(suffix=".wav") as second:
            first.write(b"one")
            second.write(b"two")
            first.flush()
            second.flush()
            batch = manager.create_batch([(first, "one.wav"), (second, "two.wav")], config)

        manager.wait(batch.job_ids[0], timeout=5)
        manager.wait(batch.job_ids[1], timeout=5)
        status = manager.get_batch(batch.id)
        assert status is not None
        assert status["status"] == "completed"
        assert status["completed"] == 2
        assert status["failed"] == 0
        assert status["progress"] == 100
    finally:
        manager.shutdown()
