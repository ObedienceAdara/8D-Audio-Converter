from __future__ import annotations

import argparse
import json
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

from spatial_audio_converter.config import AudioProcessingConfig
from spatial_audio_converter.pipeline.core import AudioPipeline


def write_fixture(path: Path, sample_rate: int, duration: float) -> None:
    frames = int(sample_rate * duration)
    t = np.arange(frames, dtype=np.float32) / sample_rate
    left = 0.2 * np.sin(2 * np.pi * 220 * t)
    right = 0.2 * np.sin(2 * np.pi * 330 * t)
    pcm = np.column_stack((left, right))
    pcm = np.round(pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one deterministic spatial-audio pipeline run.")
    parser.add_argument("--duration", type=float, default=10.0, help="Fixture duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=44100)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="spatial-benchmark-") as directory:
        root = Path(directory)
        input_path = root / "input.wav"
        output_path = root / "output.wav"
        write_fixture(input_path, args.sample_rate, args.duration)
        config = AudioProcessingConfig(output_format="wav", room_enabled=False, hrtf_enabled=False)

        started = time.perf_counter()
        artifacts = AudioPipeline().run(input_path, output_path, config)
        elapsed = time.perf_counter() - started

    print(json.dumps({
        "duration_seconds": args.duration,
        "elapsed_seconds": round(elapsed, 6),
        "realtime_factor": round(args.duration / elapsed, 3) if elapsed else None,
        "output_bytes": output_path.stat().st_size if output_path.exists() else 0,
        "renderer": artifacts.metrics.get("renderer"),
    }, indent=2))


if __name__ == "__main__":
    main()
