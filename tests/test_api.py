from pathlib import Path

import pytest

from spatial_audio_converter.api.app import create_app


class FakeRecord:
    def __init__(self, status="queued", output_path=None):
        self.id = "job-123"
        self.status = status
        self.created_at = "2026-09-06T00:00:00+00:00"
        self.started_at = None
        self.completed_at = None
        self.filename = "track.wav"
        self.batch_id = None
        self.output_path = str(output_path) if output_path else None
        self.output_format = "wav"
        self.metrics = {"peak_dbfs": -1.0}
        self.waveform = [0.1, 0.5, 0.2]
        self.progress = 100 if status == "completed" else 42 if status == "processing" else 0
        self.stage = "Complete" if status == "completed" else "Processing" if status == "processing" else "queued"
        self.error = None

    def as_dict(self):
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
            "metrics": self.metrics,
            "waveform": self.waveform,
            "progress": self.progress,
            "stage": self.stage,
            "error": self.error,
        }


class FakeJobManager:
    def __init__(self, status="queued", output_path=None):
        self.record = FakeRecord(status=status, output_path=output_path)
        self.config = None
        self.queue_depth = 0
        self.active_workers = 2
        self.batch = None

    def submit(self, _stream, filename, config, batch_id=None):
        self.record.filename = filename
        self.record.batch_id = batch_id
        self.config = config
        return self.record

    def get(self, job_id):
        return self.record if job_id == self.record.id else None

    def create_batch(self, items, config):
        self.config = config
        self.batch = {
            "id": "batch-123",
            "created_at": "2026-09-06T00:00:00+00:00",
            "status": "queued",
            "progress": 0,
            "total": len(items),
            "completed": 0,
            "failed": 0,
            "jobs": [],
        }
        return type("Batch", (), {"id": "batch-123"})()

    def get_batch(self, batch_id):
        return self.batch if batch_id == "batch-123" else None


@pytest.fixture
def client():
    manager = FakeJobManager()
    app = create_app(manager)
    app.testing = True
    return app.test_client(), manager


def test_health_endpoint(client):
    test_client, _manager = client
    response = test_client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["workers"] == 2


def test_home_page_renders(client):
    test_client, _manager = client
    response = test_client.get("/")
    assert response.status_code == 200
    assert b"Spatial Audio Converter" in response.data
    assert b"Waveform" in response.data


def test_presets_endpoint(client):
    test_client, _manager = client
    response = test_client.get("/api/presets")
    assert response.status_code == 200
    names = {item["name"] for item in response.get_json()["presets"]}
    assert names == {"balanced", "deep", "clean"}


def test_create_job_requires_file(client):
    test_client, _manager = client
    response = test_client.post("/api/jobs")
    assert response.status_code == 400
    assert "No audio file uploaded" in response.get_json()["error"]


@pytest.mark.parametrize("filename", ["track.txt", "track.exe", "track.mp4"])
def test_create_job_rejects_invalid_file_type(client, filename):
    test_client, _manager = client
    response = test_client.post(
        "/api/jobs",
        data={"file": (bytes(16), filename)},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "Unsupported input format" in response.get_json()["error"]


def test_create_job_validates_processing_parameters(client):
    test_client, _manager = client
    response = test_client.post(
        "/api/jobs",
        data={
            "file": (bytes(16), "track.wav"),
            "depth": "2.0",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "depth" in response.get_json()["error"]


def test_create_job_returns_accepted_and_uses_phase_two_defaults(client):
    test_client, manager = client
    response = test_client.post(
        "/api/jobs",
        data={"file": (bytes(16), "../../track.wav"), "output_format": "wav"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 202
    body = response.get_json()
    assert body["status_url"] == "/api/jobs/job-123"
    assert manager.config.hrtf_enabled is False
    assert manager.config.output_format == "wav"
    assert manager.record.filename == "track.wav"
    assert "progress" in body["job"]
    assert "stage" in body["job"]


def test_job_status_is_404_for_unknown_job(client):
    test_client, _manager = client
    response = test_client.get("/api/jobs/unknown")
    assert response.status_code == 404


def test_completed_job_exposes_download_preview_and_waveform_urls(tmp_path: Path):
    output = tmp_path / "job-123.wav"
    output.write_bytes(b"RIFFfake")
    manager = FakeJobManager(status="completed", output_path=output)
    app = create_app(manager)
    response = app.test_client().get("/api/jobs/job-123")
    assert response.status_code == 200
    body = response.get_json()
    assert body["download_url"] == "/api/jobs/job-123/download"
    assert body["preview_url"] == "/api/jobs/job-123/preview"
    assert body["waveform_url"] == "/api/jobs/job-123/waveform"


def test_completed_download_is_served(tmp_path: Path):
    output = tmp_path / "job-123.wav"
    payload = b"RIFFfake"
    output.write_bytes(payload)
    app = create_app(FakeJobManager(status="completed", output_path=output))
    response = app.test_client().get("/api/jobs/job-123/download")
    assert response.status_code == 200
    assert response.data == payload


def test_completed_preview_is_served(tmp_path: Path):
    output = tmp_path / "job-123.wav"
    payload = b"RIFFfake"
    output.write_bytes(payload)
    app = create_app(FakeJobManager(status="completed", output_path=output))
    response = app.test_client().get("/api/jobs/job-123/preview")
    assert response.status_code == 200
    assert response.data == payload


def test_completed_waveform_is_returned(tmp_path: Path):
    output = tmp_path / "job-123.wav"
    output.write_bytes(b"RIFFfake")
    app = create_app(FakeJobManager(status="completed", output_path=output))
    response = app.test_client().get("/api/jobs/job-123/waveform")
    assert response.status_code == 200
    assert response.get_json() == {"waveform": [0.1, 0.5, 0.2]}


def test_download_rejects_incomplete_job(client):
    test_client, _manager = client
    response = test_client.get("/api/jobs/job-123/download")
    assert response.status_code == 409


def test_batch_endpoint_accepts_multiple_files(client):
    test_client, manager = client
    response = test_client.post(
        "/api/batches",
        data={
            "files": [
                (bytes(16), "one.wav"),
                (bytes(16), "two.mp3"),
            ],
            "preset": "deep",
            "output_format": "wav",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 202
    assert response.get_json()["status_url"] == "/api/batches/batch-123"
    assert manager.config.depth == 0.95
    assert manager.batch["total"] == 2


def test_batch_status_is_404_for_unknown_batch(client):
    test_client, _manager = client
    response = test_client.get("/api/batches/unknown")
    assert response.status_code == 404
