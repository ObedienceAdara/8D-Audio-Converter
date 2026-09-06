from __future__ import annotations

import threading
from collections.abc import Callable

from .queue import ConversionTask, InProcessJobQueue

TaskHandler = Callable[[ConversionTask], None]


class ConversionWorker:
    """Dedicated worker thread that drains conversion tasks from the queue."""

    def __init__(self, queue: InProcessJobQueue[ConversionTask], handler: TaskHandler, name: str) -> None:
        self.queue = queue
        self.handler = handler
        self.name = name
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            task = self.queue.get(timeout=0.5)
            if task is None:
                continue
            try:
                self.handler(task)
            finally:
                self.queue.task_done()
