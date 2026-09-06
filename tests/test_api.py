from pathlib import Path
from types import SimpleNamespace

import pytest

from spatial_audio_converter.api.app import create_app


class FakeRecord:
    def __init__(self, status="queued", output_path=None):
        self.id = "job-123"
        self.status = status
        self.created_at = "2026-09-06T00:00:00+00:00"
        self.filename = "track.wav"
        self.output_path = str(output_path) if output_path else None
        self.metrics = {"peak_dbfs": -1.0}
        self.error = None

    def as_dict(self):
        return {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "filename": self.filename,
            "output_path": self.output_path,
            "metrics": self.metrics,
            "error": self.error,
        }


class FakeJobManager:
    def __init__(self, status="queued", output_path=None):
        self.record = FakeRecord(status=status, output_path=output_path)
        self.config = None

    def submit(self, _stream, filename, config):
        self.record.filename = filename
        self.config = config
        return self.record

    def get(self, job_id):
        return self.record if job_id == self.record.id else None


def test_health_endpoint():
    app = create_app(FakeJobManager())
    app.testing = True
    response = app.test_client().get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_home_page_renders():
    app = create_app(FakeJobManager())
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert b"Spatial Audio Converter" in response.data


def test_create_job_requires_file():
    app = create_app(FakeJobManager())
    response = app.test_client().post("/api/jobs")
    assert response.status_code == 400
    assert "No audio file uploaded" in response.get_json()["error"]


@pytest.mark.parametrize("filename", ["track.txt", "track.exe", "track.mp4"])
def test_create_job_rejects_invalid_file_type(filename):
    app = create_app(FakeJobManager())
    response = app.test_client().post(
        "/api/jobs",
        data={"file": (bytes(16), filename)},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "Unsupported input format" in response.get_json()["error"]


def test_create_job_validates_processing_parameters():
    manager = FakeJobManager()
    app = create_app(manager)
    response = app.test_client().post(
        "/api/jobs",
        data={
            "file": (bytes(16), "track.wav"),
            "depth": "2.0",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "depth" in response.get_json()["error"]


def test_create_job_returns_accepted_and_uses_phase_two_defaults():
    manager = FakeJobManager()
    app = create_app(manager)
    response = app.test_client().post(
        "/api/jobs",
        data={"file": (bytes(16), "../../track.wav")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 202
    body = response.get_json()
    assert body["status_url"] == "/api/jobs/job-123"
    assert manager.config.hrtf_enabled is False
    assert manager.config.output_format == "mp3"
    assert manager.record.filename == "track.wav"


def test_job_status_is_404_for_unknown_job():
    app = create_app(FakeJobManager())
    response = app.test_client().get("/api/jobs/unknown")
    assert response.status_code == 404


def test_completed_job_exposes_download_url(tmp_path: Path):
    output = tmp_path / "job-123.wav"
    output.write_bytes(b"RIFFfake")
    manager = FakeJobManager(status="completed", output_path=output)
    app = create_app(manager)
    response = app.test_client().get("/api/jobs/job-123")
    assert response.status_code == 200
    assert response.get_json()["download_url"] == "/api/jobs/job-123/download"
