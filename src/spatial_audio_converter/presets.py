from __future__ import annotations

from dataclasses import asdict, replace

from .config import AudioProcessingConfig

PRESET_OVERRIDES: dict[str, dict] = {
    "balanced": {
        "pan_speed_hz": 0.50,
        "depth": 0.75,
        "reverb_delay_ms": 50,
        "reverb_decay": 0.25,
        "reverb_mix": 0.25,
        "target_rms_db": -18.0,
        "limiter_db": -1.0,
    },
    "deep": {
        "pan_speed_hz": 0.35,
        "depth": 0.95,
        "reverb_delay_ms": 65,
        "reverb_decay": 0.35,
        "reverb_mix": 0.35,
        "target_rms_db": -18.0,
        "limiter_db": -1.0,
    },
    "clean": {
        "pan_speed_hz": 0.70,
        "depth": 0.65,
        "reverb_delay_ms": 35,
        "reverb_decay": 0.15,
        "reverb_mix": 0.10,
        "target_rms_db": -18.0,
        "limiter_db": -1.0,
        "room_enabled": False,
    },
}


def get_preset(name: str | None) -> AudioProcessingConfig:
    key = (name or "balanced").strip().lower()
    if key not in PRESET_OVERRIDES:
        raise ValueError(f"Unknown preset: {key}. Available: {sorted(PRESET_OVERRIDES)}")
    return replace(AudioProcessingConfig(), **PRESET_OVERRIDES[key])


def list_presets() -> list[dict]:
    return [
        {"name": name, "config": asdict(get_preset(name))}
        for name in PRESET_OVERRIDES
    ]
