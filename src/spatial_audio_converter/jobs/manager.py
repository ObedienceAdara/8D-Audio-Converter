from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

from ..config import AudioProcessingConfig
from ..pipeline.core import AudioPipeline
from ..storage.local import LocalStorage
from .queue import ConversionTask, InProcessJobQueue, QueueFullError
from .worker import ConversionWorker


@dataclass(slots=True)
class JobRecord:
    id: str
    status: str
    created_at: str
    filename: str
    output_path: str | None = None
    metrics: dict | None = None
    waveform: list[float] = field(default_factory=list)
    error: str | None = None
    progress: int = 0
    stage: str = "queued"
    batch_id: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    output_format: str = "mp3"
    done_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "filename": self.filename,
            "batch_id": self.batch_id,
            "output_path": self.output_path,
            "output_format": self.output_format,
            "metrics": self.metrics or {},
            "waveform": self.waveform,
            "progress": self.progress,
            "stage": self.stage,
            "error": self.error,
        }


@dataclass(slots=True)
class BatchRecord:
    id: str
    created_at: str
    job_ids: list[str]


class JobManager:
    """Coordinate API submissions, a bounded queue, dedicated workers, and storage."""

    def __init__(
        self,
        pipeline: AudioPipeline | None = None,
        storage: LocalStorage | None = None,
        max_workers: int = 2,
        queue_size: int = 64,
        queue: InProcessJobQueue[ConversionTask] | None = None,
    ) -> None:
        self.pipeline = pipeline or AudioPipeline()
        self.storage = storage or LocalStorage()
        self.queue = queue or InProcessJobQueue[ConversionTask](maxsize=queue_size)
        self.jobs: dict[str, JobRecord] = {}
        self.batches: dict[str, BatchRecord] = {}
        self.lock = threading.RLock()
        self.workers = [
            ConversionWorker(self.queue, self._run, f"audio-worker-{index + 1}")
            for index in range(max_workers)
        ]
        for worker in self.workers:
            worker.start()

    @property
    def queue_depth(self) -> int:
        return self.queue.qsize()

    @property
    def active_workers(self) -> int:
        return sum(worker.alive for worker in self.workers)

    def submit(
        self,
        stream: BinaryIO,
        filename: str,
        config: AudioProcessingConfig,
        batch_id: str | None = None,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex
        source_path = self.storage.stage_upload(stream, filename)
        output_path = self.storage.output_path(job_id, config.output_format)
        record = JobRecord(
            id=job_id,
            status="queued",
            created_at=datetime.now(UTC).isoformat(),
            filename=filename,
            batch_id=batch_id,
            output_format=config.output_format,
        )
        task = ConversionTask(job_id, str(source_path), str(output_path), filename, config)
        with self.lock:
            self.jobs[job_id] = record
        try:
            self.queue.put(task)
        except QueueFullError:
            with self.lock:
                self.jobs.pop(job_id, None)
            self.storage.remove(source_path)
            raise
        return record

    def create_batch(
        self,
        items: list[tuple[BinaryIO, str]],
        config: AudioProcessingConfig,
    ) -> BatchRecord:
        batch = BatchRecord(
            id=uuid.uuid4().hex,
            created_at=datetime.now(UTC).isoformat(),
            job_ids=[],
        )
        with self.lock:
            self.batches[batch.id] = batch
        for stream, filename in items:
            record = self.submit(stream, filename, config, batch_id=batch.id)
            batch.job_ids.append(record.id)
        return batch

    def _set_progress(self, record: JobRecord, progress: int, stage: str) -> None:
        with self.lock:
            record.progress = progress
            record.stage = stage
            if progress < 100:
                record.status = "processing"

    def _run(self, task: ConversionTask) -> None:
        with self.lock:
            record = self.jobs.get(task.job_id)
            if record is None:
                self.storage.remove(task.source_path)
                return
            record.status = "processing"
            record.started_at = datetime.now(UTC).isoformat()
            record.stage = "Starting worker"
            record.progress = 1

        try:
            callback = lambda progress, stage: self._set_progress(record, progress, stage)
            artifacts = self.pipeline.run(
                task.source_path,
                task.output_path,
                task.config,
                progress_callback=callback,
            )
            with self.lock:
                record.output_path = artifacts.output_path
                record.metrics = artifacts.metrics
                record.waveform = artifacts.waveform
                record.progress = 100
                record.stage = "Complete"
                record.status = "completed"
                record.completed_at = datetime.now(UTC).isoformat()
            self.storage.schedule_remove(artifacts.output_path)
        except Exception as exc:  # noqa: BLE001 - worker boundary must capture pipeline failures
            with self.lock:
                record.error = str(exc)
                record.status = "failed"
                record.stage = "Failed"
                record.completed_at = datetime.now(UTC).isoformat()
        finally:
            self.storage.remove(task.source_path)
            record.done_event.set()

    def get(self, job_id: str) -> JobRecord | None:
        with self.lock:
            return self.jobs.get(job_id)

    def get_batch(self, batch_id: str) -> dict | None:
        with self.lock:
            batch = self.batches.get(batch_id)
            if batch is None:
                return None
            jobs = [self.jobs[job_id] for job_id in batch.job_ids if job_id in self.jobs]
        terminal = [job for job in jobs if job.status in {"completed", "failed"}]
        completed = sum(job.status == "completed" for job in jobs)
        failed = sum(job.status == "failed" for job in jobs)
        if not jobs or (len(terminal) < len(jobs) and not any(job.status == "processing" for job in jobs)):
            status = "queued"
        elif len(terminal) == len(jobs):
            status = "failed" if failed else "completed"
        else:
            status = "processing"
        progress = round(sum(job.progress for job in jobs) / len(jobs), 1) if jobs else 0
        return {
            "id": batch.id,
            "created_at": batch.created_at,
            "status": status,
            "progress": progress,
            "total": len(jobs),
            "completed": completed,
            "failed": failed,
            "jobs": [job.as_dict() for job in jobs],
        }

    def wait(self, job_id: str, timeout: float | None = None) -> JobRecord | None:
        record = self.get(job_id)
        if record is None:
            return None
        record.done_event.wait(timeout)
        return record

    def shutdown(self, timeout: float = 5.0) -> None:
        for worker in self.workers:
            worker.stop()
        for worker in self.workers:
            worker.join(timeout)
