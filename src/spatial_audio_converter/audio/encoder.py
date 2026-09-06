from __future__ import annotations

import wave
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import ClassVar

import numpy as np
from pydub import AudioSegment

from ..domain.models import AudioBuffer


class AudioEncoder:
    """Encode normalized floating-point audio into WAV or MP3 artifacts."""

    SUPPORTED_OUTPUTS: ClassVar[set[str]] = {"wav", "mp3"}

    def encode(
        self,
        audio: AudioBuffer,
        output_path: str | Path,
        output_format: str,
        *,
        bitrate: str = "320k",
    ) -> Path:
        output_format = output_format.lower().lstrip(".")
        if output_format not in self.SUPPORTED_OUTPUTS:
            raise ValueError(f"Unsupported output format: {output_format}")

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        samples = np.clip(audio.samples, -1.0, 1.0)
        pcm = np.round(samples * 32767.0).astype(np.int16)

        if output_format == "wav":
            with wave.open(str(destination), "wb") as wav_file:
                wav_file.setnchannels(audio.channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(audio.sample_rate)
                wav_file.writeframes(pcm.tobytes())
            return destination

        with NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            temp_path = Path(temp.name)
        try:
            with wave.open(str(temp_path), "wb") as wav_file:
                wav_file.setnchannels(audio.channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(audio.sample_rate)
                wav_file.writeframes(pcm.tobytes())
            segment = AudioSegment.from_wav(str(temp_path))
            tags = audio.metadata.as_tags()
            segment.export(str(destination), format="mp3", bitrate=bitrate, tags=tags or None)
        finally:
            temp_path.unlink(missing_ok=True)
        return destination
