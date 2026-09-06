from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import BinaryIO


class LocalStorage:
    """Local artifact storage used by the reference deployment."""

    def __init__(self, root: str | None = None) -> None:
        self.root = Path(root or os.getenv("AUDIO_STORAGE_DIR", "/tmp/8d-audio-converter"))
        self.uploads = self.root / "uploads"
        self.outputs = self.root / "outputs"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.outputs.mkdir(parents=True, exist_ok=True)

    def stage_upload(self, stream: BinaryIO, original_name: str) -> Path:
        suffix = Path(original_name).suffix.lower() or ".bin"
        path = self.uploads / f"{uuid.uuid4().hex}{suffix}"
        with path.open("wb") as destination:
            while chunk := stream.read(1024 * 1024):
                destination.write(chunk)
        return path

    def output_path(self, job_id: str, extension: str) -> Path:
        return self.outputs / f"{job_id}.{extension.lstrip('.') }"

    def remove(self, path: str | Path) -> None:
        Path(path).unlink(missing_ok=True)

    def schedule_remove(self, path: str | Path, delay_seconds: int = 600) -> None:
        def worker() -> None:
            threading.Event().wait(delay_seconds)
            self.remove(path)
        threading.Thread(target=worker, daemon=True).start()
