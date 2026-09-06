from pathlib import Path
import wave

import numpy as np

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


def test_end_to_end_wav_conversion(tmp_path: Path):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    _write_test_wav(input_path)

    config = AudioProcessingConfig(
        output_format="wav",
        room_enabled=False,
        hrtf_enabled=False,
        max_duration_seconds=10,
    )
    artifacts = AudioPipeline().run(input_path, output_path, config)

    assert output_path.exists()
    assert Path(artifacts.output_path) == output_path
    assert artifacts.metrics["output_frames"] == artifacts.metrics["input_frames"]
    assert artifacts.metrics["clipped_samples"] == 0

    decoded = AudioDecoder().decode(output_path, max_duration_seconds=10)
    assert decoded.channels == 2
    assert decoded.sample_rate == 8000
    assert decoded.frames == int(8000 * 0.25)
    assert np.isfinite(decoded.samples).all()
