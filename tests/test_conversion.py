from pathlib import Path
import shutil
import wave

import numpy as np
import pytest

from spatial_audio_converter.audio.decoder import AudioDecoder
from spatial_audio_converter.config import AudioProcessingConfig
from spatial_audio_converter.pipeline.core import AudioPipeline


def _write_test_wav(path: Path, sample_rate: int = 8000, duration: float = 0.25) -> None:
    frames = int(sample_rate * duration)
    t = np.arange(frames, dtype=np.float32) / sample_rate
    left = 0.2 * np.sin(2 * np.pi * 440 * t)
    right = 0.2 * np.sin(2 * np.pi * 660 * t)
    pcm = np.column_stack((left, right))
    pcm = np.round(pcm * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def _config(output_format: str) -> AudioProcessingConfig:
    return AudioProcessingConfig(
        output_format=output_format,
        room_enabled=False,
        hrtf_enabled=False,
        max_duration_seconds=10,
    )


def test_end_to_end_wav_conversion(tmp_path: Path):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    _write_test_wav(input_path)

    artifacts = AudioPipeline().run(input_path, output_path, _config("wav"))

    assert output_path.exists()
    assert Path(artifacts.output_path) == output_path
    assert artifacts.metrics["output_frames"] == artifacts.metrics["input_frames"]
    assert artifacts.metrics["clipped_samples"] == 0

    decoded = AudioDecoder().decode(output_path, max_duration_seconds=10)
    assert decoded.channels == 2
    assert decoded.sample_rate == 8000
    assert decoded.frames == int(8000 * 0.25)
    assert np.isfinite(decoded.samples).all()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required for MP3 integration test")
def test_end_to_end_mp3_conversion(tmp_path: Path):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.mp3"
    _write_test_wav(input_path)

    artifacts = AudioPipeline().run(input_path, output_path, _config("mp3"))

    assert output_path.exists()
    assert Path(artifacts.output_path) == output_path
    assert output_path.stat().st_size > 0
    decoded = AudioDecoder().decode(output_path, max_duration_seconds=10)
    assert decoded.channels == 2
    assert decoded.frames > 0
    assert np.isfinite(decoded.samples).all()
