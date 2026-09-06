from pathlib import Path

import pytest

from spatial_audio_converter.api.app import create_app


class FakeRecord:
    def __init__(self, status="queued", output_path=None, report_path=None):
        self.id = "job-123"
        self.status = status
        self.created_at = "2026-09-06T00:00:00+00:00"
        self.filename = "track.wav"
        self.output_path = str(output_path) if output_path else None
        self.quality_report_path = str(report_path) if report_path else None
        self.metrics = {"peak_dbfs": -1.0}
        self.error = None
        self.progress = 0
        self.stage = "queued"

    def as_dict(self):
        return {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "filename": self.filename,
            "output_path": self.output_path,
            "metrics": self.metrics,
            "error": self.error,
            "progress": self.progress,
            "stage": self.stage,
        }


class FakeJobManager:
    def __init__(self, status="queued", output_path=None, report_path=None):
        self.record = FakeRecord(status=status, output_path=output_path, report_path=report_path)
        self.config = None
        self.queue_depth = 0
        self.active_workers = 1

    def submit(self, _stream, filename, config):
        self.record.filename = filename
        self.config = config
        return self.record

    def get(self, job_id):
        return self.record if job_id == self.record.id else None


def test_health_endpoint():
    app = create_app(FakeJobManager())
    response = app.test_client().get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "queue_depth": 0, "workers": 1}


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
        data={"file": (bytes(16), "track.wav"), "depth": "2.0"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "depth" in response.get_json()["error"]


def test_create_job_returns_accepted_and_uses_phase_three_defaults():
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
    assert manager.config.hrtf_enabled is True
    assert manager.config.spatial_mode == "binaural"
    assert manager.config.headphone_mode is True
    assert manager.config.output_format == "mp3"
    assert manager.record.filename == "track.wav"


def test_create_job_accepts_sofa_configuration():
    manager = FakeJobManager()
    app = create_app(manager)
    response = app.test_client().post(
        "/api/jobs",
        data={
            "file": (bytes(16), "track.wav"),
            "hrtf_source": "sofa",
            "hrtf_sofa_path": "/tmp/listener.sofa",
            "room_model": "measured-wav",
            "room_ir_path": "/tmp/room.wav",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 202
    assert manager.config.hrtf_source == "sofa"
    assert manager.config.hrtf_sofa_path == "/tmp/listener.sofa"
    assert manager.config.room_model == "measured-wav"


def test_job_status_is_404_for_unknown_job():
    app = create_app(FakeJobManager())
    response = app.test_client().get("/api/jobs/unknown")
    assert response.status_code == 404


def test_completed_job_exposes_download_preview_and_report_urls(tmp_path: Path):
    output = tmp_path / "job-123.wav"
    report = tmp_path / "job-123.wav.quality.json"
    output.write_bytes(b"RIFFfake")
    report.write_text("{}", encoding="utf-8")
    manager = FakeJobManager(status="completed", output_path=output, report_path=report)
    app = create_app(manager)
    response = app.test_client().get("/api/jobs/job-123")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["download_url"] == "/api/jobs/job-123/download"
    assert payload["preview_url"] == "/api/jobs/job-123/preview"
    assert payload["report_url"] == "/api/jobs/job-123/report"


def test_completed_download_is_served(tmp_path: Path):
    output = tmp_path / "job-123.wav"
    output.write_bytes(b"RIFFfake")
    app = create_app(FakeJobManager(status="completed", output_path=output))
    response = app.test_client().get("/api/jobs/job-123/download")
    assert response.status_code == 200
    assert response.data == b"RIFFfake"


def test_completed_preview_is_served(tmp_path: Path):
    output = tmp_path / "job-123.wav"
    output.write_bytes(b"RIFFfake")
    app = create_app(FakeJobManager(status="completed", output_path=output))
    response = app.test_client().get("/api/jobs/job-123/preview")
    assert response.status_code == 200
    assert response.data == b"RIFFfake"


def test_completed_report_is_served(tmp_path: Path):
    output = tmp_path / "job-123.wav"
    report = tmp_path / "report.json"
    output.write_bytes(b"RIFFfake")
    report.write_text('{"report_version": 1}', encoding="utf-8")
    app = create_app(FakeJobManager(status="completed", output_path=output, report_path=report))
    response = app.test_client().get("/api/jobs/job-123/report")
    assert response.status_code == 200
    assert response.json["report_version"] == 1


def test_download_rejects_incomplete_job():
    app = create_app(FakeJobManager(status="processing"))
    response = app.test_client().get("/api/jobs/job-123/download")
    assert response.status_code == 409
