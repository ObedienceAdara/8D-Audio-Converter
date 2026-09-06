from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from ..config import AudioProcessingConfig
from ..jobs.manager import JobManager


MAX_UPLOAD_BYTES = 32 * 1024 * 1024
ALLOWED_INPUTS = {"mp3", "wav", "flac", "ogg", "m4a", "aac"}


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
            config = AudioProcessingConfig(
                pan_speed_hz=float(request.form.get("pan_speed_hz", 0.5)),
                depth=float(request.form.get("depth", 0.95)),
                reverb_delay_ms=int(request.form.get("reverb_delay_ms", 50)),
                reverb_decay=float(request.form.get("reverb_decay", 0.3)),
                reverb_mix=float(request.form.get("reverb_mix", 0.3)),
                target_rms_db=float(request.form.get("target_rms_db", -18.0)),
                limiter_db=float(request.form.get("limiter_db", -1.0)),
                output_format=request.form.get("output_format", "mp3"),
                output_bitrate=request.form.get("output_bitrate", "320k"),
                room_enabled=request.form.get("room_enabled", "true").lower() == "true",
                hrtf_enabled=request.form.get("hrtf_enabled", "true").lower() == "true",
            )
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

        record = manager.submit(uploaded.stream, filename, config)
        return jsonify({"job": record.as_dict(), "status_url": f"/api/jobs/{record.id}"}), 202

    @app.get("/api/jobs/<job_id>")
    def get_job(job_id: str):
        record = manager.get(job_id)
        if record is None:
            return jsonify({"error": "Job not found."}), 404
        body = record.as_dict()
        if record.status == "completed":
            body["download_url"] = f"/api/jobs/{record.id}/download"
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
        return send_file(
            output,
            as_attachment=True,
            download_name=f"{Path(record.filename).stem}_spatial.{output.suffix.lstrip('.')}",
        )

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_exc):
        return jsonify({"error": "File exceeds the 32 MB upload limit."}), 413

    return app
