from pathlib import Path

import pytest

from spatial_audio_converter.audio.decoder import AudioDecoder


@pytest.mark.parametrize("filename", ["track.txt", "track", "track.exe", "track.mp4"])
def test_decoder_rejects_unsupported_extensions(tmp_path: Path, filename: str):
    path = tmp_path / filename
    path.write_bytes(b"not audio")
    with pytest.raises(ValueError, match="Unsupported audio format"):
        AudioDecoder().decode(path)


def test_decoder_reports_missing_supported_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        AudioDecoder().decode(tmp_path / "missing.wav")


def test_decoder_enforces_duration_limit(tmp_path: Path, monkeypatch):
    path = tmp_path / "long.wav"
    path.write_bytes(b"placeholder")

    class FakeAudio:
        duration_seconds = 11.0

    monkeypatch.setattr(
        "spatial_audio_converter.audio.decoder.AudioSegment.from_file",
        lambda _path: FakeAudio(),
    )

    with pytest.raises(ValueError, match="duration exceeds"):
        AudioDecoder().decode(path, max_duration_seconds=10)
