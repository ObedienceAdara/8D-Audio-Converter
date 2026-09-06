from __future__ import annotations

from pathlib import Path

import numpy as np
from pydub import AudioSegment

from ..domain.models import AudioBuffer, AudioMetadata


class AudioDecoder:
    """Decode supported media into the pipeline's normalized representation."""

    SUPPORTED_SUFFIXES = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}

    def decode(self, input_path: str | Path, max_duration_seconds: int = 900) -> AudioBuffer:
        path = Path(input_path)
        if path.suffix.lower() not in self.SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported audio format: {path.suffix or 'unknown'}")
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        audio = AudioSegment.from_file(path)
        if audio.duration_seconds > max_duration_seconds:
            raise ValueError(
                f"Audio duration exceeds the {max_duration_seconds}-second limit."
            )

        samples = np.array(audio.get_array_of_samples())
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels))
        else:
            samples = samples.reshape((-1, 1))

        max_int = float(1 << (8 * audio.sample_width - 1))
        normalized = samples.astype(np.float32) / max_int
        metadata = AudioMetadata()
        if audio.tags:
            metadata = AudioMetadata(
                title=audio.tags.get("title", ""),
                artist=audio.tags.get("artist", ""),
                album=audio.tags.get("album", ""),
                genre=audio.tags.get("genre", ""),
                year=audio.tags.get("date", audio.tags.get("year", "")),
                comment=audio.tags.get("comment", ""),
            )
        return AudioBuffer(normalized, audio.frame_rate, metadata)
