from __future__ import annotations

import html
import json
from pathlib import Path


def write_quality_report(report: dict, output_path: str | Path) -> str:
    """Persist JSON and a small human-readable HTML quality report."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return str(destination)


def write_quality_html(report: dict, output_path: str | Path) -> str:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    context = report.get("context", {})
    before = report.get("before", {})
    after = report.get("after", {})
    comparison = report.get("comparison", {})
    rows = [
        ("Integrated LUFS", before.get("integrated_lufs"), after.get("integrated_lufs")),
        ("RMS dBFS", before.get("rms_dbfs"), after.get("rms_dbfs")),
        ("Peak dBFS", before.get("peak_dbfs"), after.get("peak_dbfs")),
        ("True peak dBFS", before.get("true_peak_dbfs"), after.get("true_peak_dbfs")),
        ("Crest factor dB", before.get("crest_factor_db"), after.get("crest_factor_db")),
        ("Spectral centroid Hz", before.get("spectral", {}).get("spectral_centroid_hz"), after.get("spectral", {}).get("spectral_centroid_hz")),
        ("Spectral rolloff Hz", before.get("spectral", {}).get("spectral_rolloff_hz"), after.get("spectral", {}).get("spectral_rolloff_hz")),
    ]
    table = "".join(
        f"<tr><td>{html.escape(str(label))}</td><td>{html.escape(str(left))}</td><td>{html.escape(str(right))}</td></tr>"
        for label, left, right in rows
    )
    destination.write_text(
        "<html><head><meta charset='utf-8'><title>Spatial Audio Quality Report</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:960px;margin:40px auto;padding:0 20px}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:8px;text-align:left}pre{white-space:pre-wrap;background:#f5f5f5;padding:12px}</style></head><body>"
        f"<h1>Spatial Audio Quality Report</h1><p>Context: {html.escape(json.dumps(context))}</p>"
        "<table><thead><tr><th>Metric</th><th>Before</th><th>After</th></tr></thead>"
        f"<tbody>{table}</tbody></table>"
        f"<h2>Before/after comparison</h2><pre>{html.escape(json.dumps(comparison, indent=2, allow_nan=False))}</pre>"
        "</body></html>",
        encoding="utf-8",
    )
    return str(destination)
