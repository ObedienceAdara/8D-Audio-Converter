from pathlib import Path

import h5py
import numpy as np

from spatial_audio_converter.effects.reverb import RoomReverb
from spatial_audio_converter.processing.quality import QualityAnalyzer
from spatial_audio_converter.processing.report import write_quality_html, write_quality_report
from spatial_audio_converter.spatial.hrtf import HRTFConvolver, HRTFDatabase


def test_synthetic_hrtf_is_real_fir_convolution_and_directional():
    database = HRTFDatabase.synthetic(sample_rate=48000, ir_length=128)
    assert database.ir.shape == (13, 2, 128)
    impulse = np.zeros(1024, dtype=np.float32)
    impulse[0] = 1.0
    left_output = HRTFConvolver(ir_length=128, block_size=128).process(
        impulse, np.full(len(impulse), -90.0, dtype=np.float32), 48000
    )
    right_output = HRTFConvolver(ir_length=128, block_size=128).process(
        impulse, np.full(len(impulse), 90.0, dtype=np.float32), 48000
    )
    assert left_output.shape == right_output.shape == (1024, 2)
    assert np.sum(np.abs(left_output[:, 0])) > np.sum(np.abs(left_output[:, 1]))
    assert np.sum(np.abs(right_output[:, 1])) > np.sum(np.abs(right_output[:, 0]))
    assert np.count_nonzero(np.abs(left_output) > 1e-8) > 1


def test_sofa_loader_accepts_two_receiver_hrir(tmp_path: Path):
    path = tmp_path / "listener.sofa"
    with h5py.File(path, "w") as sofa:
        sofa.attrs["SOFAConventions"] = "SimpleFreeFieldHRIR"
        sofa.create_dataset(
            "Data.IR",
            data=np.stack([
                np.stack([np.array([1.0, 0.5, 0.0]), np.array([0.9, 0.2, 0.0])]),
                np.stack([np.array([0.8, 0.2, 0.0]), np.array([1.0, 0.4, 0.0])]),
            ]).astype(np.float32),
        )
        sofa.create_dataset("Data.SamplingRate", data=np.array([48000.0]))
        sofa.create_dataset("SourcePosition", data=np.array([[-90.0, 0.0, 1.0], [90.0, 0.0, 1.0]]))
    database = HRTFDatabase.from_sofa(path)
    assert database.ir.shape == (2, 2, 3)
    assert database.source.startswith("sofa:")
    selected = database.nearest_ir(90.0, 48000)
    assert np.allclose(selected[1, :2], [1.0, 0.4])


def test_room_model_adds_early_reflection_and_late_tail():
    source = np.zeros((4096, 2), dtype=np.float32)
    source[0] = 1.0
    room = RoomReverb()
    output = room.process(source, 48000, 40, 0.6, 1.0, room_size=0.8, damping=0.3)
    assert output.shape == source.shape
    assert np.count_nonzero(np.abs(output) > 1e-7) > 20
    assert room.last_diagnostics["model"] == "schroeder-moorer"


def test_measured_room_ir_can_replace_algorithmic_model(tmp_path: Path):
    import wave

    path = tmp_path / "room.wav"
    pcm = np.array([32767, 0, 16384, 0], dtype=np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(pcm.tobytes())
    source = np.zeros((16, 2), dtype=np.float32)
    source[0] = 1.0
    output = RoomReverb(path).process(source, 48000, 50, 0.5, 1.0, room_model="measured-wav")
    assert np.count_nonzero(output) >= 2


def test_quality_analyzer_reports_requested_measurements():
    sample_rate = 48000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    before = (0.1 * np.sin(2 * np.pi * 1000 * time))[:, None]
    after = np.column_stack((before[:, 0], before[:, 0] * 0.8))
    analyzer = QualityAnalyzer()
    report = analyzer.automated_report(before, after, sample_rate, {"test": True})
    assert "integrated_lufs" in report["before"]
    assert "true_peak_dbfs" in report["after"]
    assert "crest_factor_db" in report["after"]
    assert "spectral" in report["after"]
    assert "downmix_snr_db" in report["comparison"]
    assert "band_level_delta_db" in report["comparison"]
    assert report["context"]["test"] is True


def test_quality_report_writes_json_and_html(tmp_path: Path):
    analyzer = QualityAnalyzer()
    audio = np.zeros((20000, 1), dtype=np.float32)
    report = analyzer.automated_report(audio, audio, 48000)
    json_path = write_quality_report(report, tmp_path / "result.quality.json")
    html_path = write_quality_html(report, tmp_path / "result.quality.html")
    assert Path(json_path).exists()
    assert Path(html_path).exists()
