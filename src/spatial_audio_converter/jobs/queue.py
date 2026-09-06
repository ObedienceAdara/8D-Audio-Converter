from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Generic, TypeVar

from ..config import AudioProcessingConfig

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ConversionTask:
    """Immutable unit of work handed from the API-facing manager to a worker."""

    job_id: str
    source_path: str
    output_path: str
    filename: str
    config: AudioProcessingConfig


class QueueFullError(RuntimeError):
    """Raised when the bounded conversion queue cannot accept another task."""


class InProcessJobQueue(Generic[T]):
    """Bounded FIFO queue used by the reference single-process deployment."""

    def __init__(self, maxsize: int = 64) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.maxsize = maxsize
        self._queue: Queue[T] = Queue(maxsize=maxsize)

    def put(self, item: T) -> None:
        try:
            self._queue.put_nowait(item)
        except Full as exc:
            raise QueueFullError("Conversion queue is full.") from exc

    def get(self, timeout: float = 0.5) -> T | None:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def task_done(self) -> None:
        self._queue.task_done()

    def qsize(self) -> int:
        return self._queue.qsize()

    def join(self) -> None:
        self._queue.join()
