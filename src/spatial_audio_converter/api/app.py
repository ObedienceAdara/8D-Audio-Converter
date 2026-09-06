from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from ..config import AudioProcessingConfig
from ..jobs.manager import JobManager

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
    return AudioProcessingConfig(
        pan_speed_hz=float(form.get("pan_speed_hz", 0.5)),
        depth=float(form.get("depth", 0.95)),
        reverb_delay_ms=int(form.get("reverb_delay_ms", 50)),
        reverb_decay=float(form.get("reverb_decay", 0.3)),
        reverb_mix=float(form.get("reverb_mix", 0.3)),
        room_size=float(form.get("room_size", 0.7)),
        room_damping=float(form.get("room_damping", 0.35)),
        room_model=form.get("room_model", "schroeder-moorer"),
        room_ir_path=form.get("room_ir_path") or None,
        room_enabled=_parse_bool(form.get("room_enabled", "true")),
        target_rms_db=float(form.get("target_rms_db", -18.0)),
        limiter_db=float(form.get("limiter_db", -1.0)),
        output_format=form.get("output_format", "mp3"),
        output_bitrate=form.get("output_bitrate", "320k"),
        spatial_mode=form.get("spatial_mode", "binaural"),
        hrtf_enabled=_parse_bool(form.get("hrtf_enabled", "true")),
        headphone_mode=_parse_bool(form.get("headphone_mode", "true")),
        hrtf_source=form.get("hrtf_source", "synthetic"),
        hrtf_sofa_path=form.get("hrtf_sofa_path") or None,
    )


def create_app(job_manager: JobManager | None = None) -> Flask:
    app = Flask(__name__, template_folder="../web/templates", static_folder="../web/static")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    manager = job_manager or JobManager(max_workers=2)

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/jobs")
    def create_job():
        if "file" not in request.files:
            return jsonify({"error": "No audio file uploaded."}), 400
        uploaded = request.files["file"]
        filename = secure_filename(uploaded.filename or "")
        suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if not filename or suffix not in ALLOWED_INPUTS:
            return jsonify({"error": f"Unsupported input format. Allowed: {sorted(ALLOWED_INPUTS)}"}), 400
        try:
            config = _config_from_form(request.form)
            record = manager.submit(uploaded.stream, filename, config)
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"job": record.as_dict(), "status_url": f"/api/jobs/{record.id}"}), 202

    @app.get("/api/jobs/<job_id>")
    def get_job(job_id: str):
        record = manager.get(job_id)
        if record is None:
            return jsonify({"error": "Job not found."}), 404
        body = record.as_dict()
        if record.status == "completed":
            body["download_url"] = f"/api/jobs/{record.id}/download"
            body["preview_url"] = f"/api/jobs/{record.id}/preview"
            body["report_url"] = f"/api/jobs/{record.id}/report"
            body["waveform_url"] = f"/api/jobs/{record.id}/waveform"
        return jsonify(body)

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
        return send_file(output, as_attachment=True, download_name=f"{Path(record.filename).stem}_spatial{output.suffix}")

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

    @app.get("/api/jobs/<job_id>/waveform")
    def waveform_job(job_id: str):
        record = manager.get(job_id)
        if record is None:
            return jsonify({"error": "Job not found."}), 404
        if record.status != "completed":
            return jsonify({"error": "Waveform is not ready."}), 409
        return jsonify({"waveform": record.waveform})

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_exc):
        return jsonify({"error": "File exceeds the 32 MB upload limit."}), 413

    return app
