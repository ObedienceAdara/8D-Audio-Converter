from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from ..config import AudioProcessingConfig
from ..jobs.manager import JobManager, JobRecord
from ..jobs.queue import QueueFullError
from ..presets import get_preset, list_presets

MAX_UPLOAD_BYTES = 32 * 1024 * 1024
ALLOWED_INPUTS = {"mp3", "wav", "flac", "ogg", "m4a", "aac"}


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError("boolean values must be true or false")


def _config_from_form(form) -> AudioProcessingConfig:
    config = get_preset(form.get("preset", "balanced"))
    overrides = {}
    casts = {
        "pan_speed_hz": float,
        "depth": float,
        "reverb_delay_ms": int,
        "reverb_decay": float,
        "reverb_mix": float,
        "room_size": float,
        "room_damping": float,
        "room_model": str,
        "room_ir_path": str,
        "target_rms_db": float,
        "limiter_db": float,
        "output_format": str,
        "output_bitrate": str,
        "max_duration_seconds": int,
        "room_enabled": _parse_bool,
        "spatial_mode": str,
        "hrtf_enabled": _parse_bool,
        "headphone_mode": _parse_bool,
        "hrtf_source": str,
        "hrtf_sofa_path": str,
        "hrtf_interpolation_quality": str,
        "hrtf_interpolation_neighbors": int,
        "hrtf_filter_crossfade_blocks": int,
        "hrtf_trajectory_smoothing": float,
    }
    for key, caster in casts.items():
        raw = form.get(key)
        if raw not in (None, ""):
            overrides[key] = caster(raw)
    if overrides.get("room_ir_path") == "":
        overrides["room_ir_path"] = None
    if overrides.get("hrtf_sofa_path") == "":
        overrides["hrtf_sofa_path"] = None
    return replace(config, **overrides) if overrides else config


def _filename_or_error(filename: str | None) -> tuple[str | None, str | None]:
    safe_name = secure_filename(filename or "")
    suffix = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""
    if not safe_name or suffix not in ALLOWED_INPUTS:
        return None, f"Unsupported input format. Allowed: {sorted(ALLOWED_INPUTS)}"
    return safe_name, None


def _job_payload(record: JobRecord, queue_depth: int) -> dict:
    body = record.as_dict()
    body["queue_depth"] = queue_depth
    if record.status == "completed":
        body["download_url"] = f"/api/jobs/{record.id}/download"
        body["preview_url"] = f"/api/jobs/{record.id}/preview"
        body["waveform_url"] = f"/api/jobs/{record.id}/waveform"
        if record.quality_report_path:
            body["report_url"] = f"/api/jobs/{record.id}/report"
        if record.quality_report_html_path:
            body["report_html_url"] = f"/api/jobs/{record.id}/report.html"
    return body


def create_app(job_manager: JobManager | None = None) -> Flask:
    app = Flask(__name__, template_folder="../web/templates", static_folder="../web/static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    manager = job_manager or JobManager(
        max_workers=int(os.getenv("AUDIO_WORKERS", "2")),
        queue_size=int(os.getenv("AUDIO_QUEUE_SIZE", "64")),
    )
    app.extensions["job_manager"] = manager

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "queue_depth": manager.queue_depth, "workers": manager.active_workers})

    @app.get("/api/presets")
    def presets():
        return jsonify({"presets": list_presets()})

    @app.post("/api/jobs")
    def create_job():
        if "file" not in request.files:
            return jsonify({"error": "No audio file uploaded."}), 400
        uploaded = request.files["file"]
        filename, filename_error = _filename_or_error(uploaded.filename)
        if filename_error:
            return jsonify({"error": filename_error}), 400
        try:
            record = manager.submit(uploaded.stream, filename, _config_from_form(request.form))
        except QueueFullError as exc:
            return jsonify({"error": str(exc), "retry_after_seconds": 2}), 503
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"job": _job_payload(record, manager.queue_depth), "status_url": f"/api/jobs/{record.id}"}), 202

    @app.post("/api/batches")
    def create_batch():
        uploaded_files = request.files.getlist("files") or request.files.getlist("file")
        if not uploaded_files:
            return jsonify({"error": "No audio files uploaded."}), 400
        items = []
        for uploaded in uploaded_files:
            filename, filename_error = _filename_or_error(uploaded.filename)
            if filename_error:
                return jsonify({"error": filename_error}), 400
            items.append((uploaded.stream, filename))
        try:
            batch = manager.create_batch(items, _config_from_form(request.form))
        except QueueFullError as exc:
            return jsonify({"error": str(exc), "retry_after_seconds": 2}), 503
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"batch": manager.get_batch(batch.id), "status_url": f"/api/batches/{batch.id}"}), 202

    @app.get("/api/jobs/<job_id>")
    def get_job(job_id: str):
        record = manager.get(job_id)
        if record is None:
            return jsonify({"error": "Job not found."}), 404
        return jsonify(_job_payload(record, manager.queue_depth))

    @app.get("/api/batches/<batch_id>")
    def get_batch(batch_id: str):
        payload = manager.get_batch(batch_id)
        if payload is None:
            return jsonify({"error": "Batch not found."}), 404
        return jsonify(payload)

    @app.get("/api/jobs/<job_id>/download")
    def download_job(job_id: str):
        record = manager.get(job_id)
        if record is None:
            return jsonify({"error": "Job not found."}), 404
        if record.status != "completed" or not record.output_path:
            return jsonify({"error": "Output is not ready."}), 409
        output = Path(record.output_path)
        if not output.exists():
            return jsonify({"error": "Output artifact has expired."}), 410
        return send_file(output, as_attachment=True, download_name=f"{Path(record.filename).stem}_spatial.{output.suffix.lstrip('.')}")

    @app.get("/api/jobs/<job_id>/preview")
    def preview_job(job_id: str):
        record = manager.get(job_id)
        if record is None:
            return jsonify({"error": "Job not found."}), 404
        if record.status != "completed" or not record.output_path:
            return jsonify({"error": "Output is not ready."}), 409
        output = Path(record.output_path)
        if not output.exists():
            return jsonify({"error": "Output artifact has expired."}), 410
        return send_file(output, as_attachment=False, conditional=True)

    @app.get("/api/jobs/<job_id>/waveform")
    def waveform_job(job_id: str):
        record = manager.get(job_id)
        if record is None:
            return jsonify({"error": "Job not found."}), 404
        if record.status != "completed":
            return jsonify({"error": "Waveform is not ready."}), 409
        return jsonify({"waveform": record.waveform})

    @app.get("/api/jobs/<job_id>/report")
    def report_job(job_id: str):
        record = manager.get(job_id)
        if record is None:
            return jsonify({"error": "Job not found."}), 404
        if record.status != "completed" or not record.quality_report_path:
            return jsonify({"error": "Quality report is not ready."}), 409
        report = Path(record.quality_report_path)
        if not report.exists():
            return jsonify({"error": "Quality report has expired."}), 410
        return send_file(report, mimetype="application/json", as_attachment=True, download_name=f"{job_id}.quality.json")

    @app.get("/api/jobs/<job_id>/report.html")
    def report_html_job(job_id: str):
        record = manager.get(job_id)
        if record is None:
            return jsonify({"error": "Job not found."}), 404
        if record.status != "completed" or not record.quality_report_html_path:
            return jsonify({"error": "HTML quality report is not ready."}), 409
        report = Path(record.quality_report_html_path)
        if not report.exists():
            return jsonify({"error": "HTML quality report has expired."}), 410
        return send_file(report, mimetype="text/html", as_attachment=False)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_exc):
        return jsonify({"error": "File exceeds the 32 MB upload limit."}), 413

    return app
