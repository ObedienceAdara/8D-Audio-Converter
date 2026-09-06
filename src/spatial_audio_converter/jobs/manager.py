from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import threading
import uuid

from ..config import AudioProcessingConfig
from ..pipeline.core import AudioPipeline
from ..storage.local import LocalStorage


@dataclass(slots=True)
class JobRecord:
    id: str
    status: str
    created_at: str
    filename: str
    output_path: str | None = None
    metrics: dict | None = None
    error: str | None = None
    future: Future | None = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "filename": self.filename,
            "output_path": self.output_path,
            "metrics": self.metrics or {},
            "error": self.error,
        }


class JobManager:
    """Manage bounded in-process conversion jobs.

    The interface is intentionally independent of the execution backend so a
    Redis/Celery/RQ worker pool can replace this implementation later.
    """

    def __init__(self, pipeline: AudioPipeline | None = None, storage: LocalStorage | None = None, max_workers: int = 2) -> None:
        self.pipeline = pipeline or AudioPipeline()
        self.storage = storage or LocalStorage()
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="audio-worker")
        self.jobs: dict[str, JobRecord] = {}
        self.lock = threading.Lock()

    def submit(self, stream, filename: str, config: AudioProcessingConfig) -> JobRecord:
        job_id = uuid.uuid4().hex
        source_path = self.storage.stage_upload(stream, filename)
        output_path = self.storage.output_path(job_id, config.output_format)
        record = JobRecord(
            id=job_id,
            status="queued",
            created_at=datetime.now(timezone.utc).isoformat(),
            filename=filename,
        )
        with self.lock:
            self.jobs[job_id] = record
        future = self.executor.submit(self._run, record, source_path, output_path, config)
        record.future = future
        return record

    def _run(self, record: JobRecord, source_path: Path, output_path: Path, config: AudioProcessingConfig) -> None:
        with self.lock:
            record.status = "processing"
        try:
            artifacts = self.pipeline.run(source_path, output_path, config)
            record.output_path = artifacts.output_path
            record.metrics = artifacts.metrics
            record.status = "completed"
            self.storage.schedule_remove(artifacts.output_path)
        except Exception as exc:
            record.error = str(exc)
            record.status = "failed"
        finally:
            self.storage.remove(source_path)

    def get(self, job_id: str) -> JobRecord | None:
        with self.lock:
            return self.jobs.get(job_id)
