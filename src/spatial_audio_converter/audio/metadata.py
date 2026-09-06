from __future__ import annotations

from pathlib import Path

from ..domain.models import AudioBuffer


class MetadataExtractor:
    """Extract and normalize metadata into pipeline-friendly tags."""

    def extract(self, audio: AudioBuffer, source_path: str | Path) -> dict[str, str]:
        metadata = audio.metadata.as_tags()
        metadata["source_name"] = Path(source_path).name
        metadata["duration_seconds"] = f"{audio.duration_seconds:.3f}"
        metadata["sample_rate"] = str(audio.sample_rate)
        metadata["channels"] = str(audio.channels)
        return metadata
