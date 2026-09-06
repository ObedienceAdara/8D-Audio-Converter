from pathlib import Path

import h5py
import numpy as np

from spatial_audio_converter.effects.reverb import RoomReverb
from spatial_audio_converter.processing.quality import QualityAnalyzer
from spatial_audio_converter.processing.report import write_quality_html, write_quality_report
from spatial_audio_converter.spatial.hrtf import HRTFConvolver, HRTFDatabase


def test_synthetic_hrtf_is_directional_fir_convolution():
    database = HRTFDatabase.synthetic(sample_rate=48_000, ir_length=128)
    assert database.ir.shape == (13, 2, 128)

    impulse = np.zeros(1024, dtype=np.float32)
    impulse[0] = 1.0
    left_output = HRTFConvolver(ir_length=128, block_size=128, trajectory_smoothing=0.0).process(
        impulse, np.full(len(impulse), -90.0, dtype=np.float32), 48_000
    )
    right_output = HRTFConvolver(ir_length=128, block_size=128, trajectory_smoothing=0.0).process(
        impulse, np.full(len(impulse), 90.0, dtype=np.float32), 48_000
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
            data=np.stack(
                [
                    np.stack([np.array([1.0, 0.5, 0.0]), np.array([0.9, 0.2, 0.0])]),
                    np.stack([np.array([0.8, 0.2, 0.0]), np.array([1.0, 0.4, 0.0])]),
                ]
            ).astype(np.float32),
        )
        sofa.create_dataset("Data.SamplingRate", data=np.array([48_000.0]))
        sofa.create_dataset("SourcePosition", data=np.array([[-90.0, 0.0, 1.0], [90.0, 0.0, 1.0]]))

    database = HRTFDatabase.from_sofa(path)
    assert database.ir.shape == (2, 2, 3)
    assert database.source.startswith("sofa:")
    selected = database.nearest_ir(90.0, 48_000)
    assert np.allclose(selected[1, :2], [1.0, 0.4])


def test_spherical_interpolation_blends_neighboring_hrirs():
    directions = np.array([[-30.0, 0.0], [30.0, 0.0]])
    ir = np.zeros((2, 2, 16), dtype=np.float32)
    ir[0, :, 2] = [1.0, 0.2]
    ir[1, :, 2] = [0.2, 1.0]
    database = HRTFDatabase(48_000, directions[:, 0], directions[:, 1], ir, "fixture")

    midpoint = database.interpolate_ir(0.0, 0.0, 48_000, quality="spherical", neighbors=2)

    assert midpoint.shape == (2, 16)
    assert np.isclose(midpoint[0, 2], midpoint[1, 2], atol=1e-6)
    assert midpoint[0, 2] > 0.55
    assert midpoint[1, 2] > 0.55


def test_bilinear_interpolation_uses_four_3d_grid_corners():
    azimuths = np.array([-30.0, 30.0, -30.0, 30.0])
    elevations = np.array([-30.0, -30.0, 30.0, 30.0])
    ir = np.zeros((4, 2, 8), dtype=np.float32)
    ir[:, 0, 1] = np.array([1.0, 3.0, 5.0, 7.0])
    ir[:, 1, 1] = np.array([7.0, 5.0, 3.0, 1.0])
    database = HRTFDatabase(48_000, azimuths, elevations, ir, "grid")

    result = database.interpolate_ir(0.0, 0.0, 48_000, quality="bilinear", neighbors=4)

    assert np.isclose(result[0, 1], 4.0, atol=1e-5)
    assert np.isclose(result[1, 1], 4.0, atol=1e-5)


def test_spherical_interpolation_wraps_continuously_at_azimuth_seam():
    azimuths = np.array([-179.0, 179.0])
    elevations = np.zeros(2)
    ir = np.zeros((2, 2, 8), dtype=np.float32)
    ir[0, :, 0] = [1.0, 0.3]
    ir[1, :, 0] = [0.9, 0.4]
    database = HRTFDatabase(48_000, azimuths, elevations, ir, "seam")

    left = database.interpolate_ir(179.5, 0.0, 48_000, quality="spherical", neighbors=2)
    right = database.interpolate_ir(-179.5, 0.0, 48_000, quality="spherical", neighbors=2)

    assert np.allclose(left, right, atol=0.02)


def test_hrtf_filter_crossfade_reduces_directional_jump():
    convolver = HRTFConvolver(
        ir_length=128,
        block_size=128,
        interpolation_quality="nearest",
        filter_crossfade_blocks=2,
        trajectory_smoothing=0.0,
    )
    first = convolver._interpolated_filter(-60.0, 0.0, 48_000)
    second = convolver._interpolated_filter(60.0, 0.0, 48_000)
    halfway = convolver._crossfade_filter(first, second, 0.5)

    assert np.allclose(halfway, 0.5 * (first + second), atol=1e-7)
    assert np.linalg.norm(halfway - first) < np.linalg.norm(second - first)
    assert np.linalg.norm(halfway - second) < np.linalg.norm(second - first)


def test_hrtf_trajectory_smoothing_is_wrap_aware():
    convolver = HRTFConvolver(ir_length=128, trajectory_smoothing=0.5)
    trajectory = np.array([179.0, -179.0, -178.0], dtype=np.float32)

    smoothed = convolver._smooth_trajectory(trajectory)

    step = np.abs(((smoothed[1:] - smoothed[:-1] + 180.0) % 360.0) - 180.0)
    assert np.max(step) < 5.0


def test_hrtf_process_accepts_elevation_trajectory():
    convolver = HRTFConvolver(ir_length=128, block_size=64, trajectory_smoothing=0.0)
    impulse = np.zeros(256, dtype=np.float32)
    impulse[0] = 1.0
    azimuth = np.zeros_like(impulse)
    elevation = np.linspace(-30.0, 30.0, len(impulse), dtype=np.float32)

    output = convolver.process(impulse, azimuth, 48_000, elevation)

    assert output.shape == (256, 2)
    assert np.isfinite(output).all()


def test_room_model_has_reflections_and_late_tail():
    source = np.zeros((4096, 2), dtype=np.float32)
    source[0] = 1.0
    room = RoomReverb()
    output = room.process(source, 48_000, 40, 0.6, 1.0, room_size=0.8, damping=0.3)
    assert output.shape == source.shape
    assert np.count_nonzero(np.abs(output) > 1e-7) > 20
    assert room.last_diagnostics["model"] == "schroeder-moorer"


def test_measured_room_ir_replaces_algorithmic_room(tmp_path: Path):
    import wave

    path = tmp_path / "room.wav"
    pcm = np.array([32767, 0, 16384, 0], dtype=np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48_000)
        wav.writeframes(pcm.tobytes())

    source = np.zeros((16, 2), dtype=np.float32)
    source[0] = 1.0
    output = RoomReverb(path).process(source, 48_000, 50, 0.5, 1.0, room_model="measured-wav")
    assert np.count_nonzero(output) >= 2


def test_quality_report_contains_loudness_dynamics_spectral_and_snr():
    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    before = (0.1 * np.sin(2 * np.pi * 1_000 * time))[:, None]
    after = np.column_stack((before[:, 0], before[:, 0] * 0.8))

    report = QualityAnalyzer().automated_report(before, after, sample_rate, {"test": True})
    assert "integrated_lufs" in report["before"]
    assert "rms_dbfs" in report["after"]
    assert "peak_dbfs" in report["after"]
    assert "true_peak_dbfs" in report["after"]
    assert "crest_factor_db" in report["after"]
    assert "frequency_response" in report["after"]
    assert "snr_db" in report["comparison"]
    assert "spectral_distance_rmse_db" in report["comparison"]
    assert report["context"]["test"] is True


def test_quality_reports_are_persisted_without_nan_or_inf(tmp_path: Path):
    audio = np.zeros((20_000, 1), dtype=np.float32)
    report = QualityAnalyzer().automated_report(audio, audio, 48_000)
    json_path = write_quality_report(report, tmp_path / "result.quality.json")
    html_path = write_quality_html(report, tmp_path / "result.quality.html")
    assert Path(json_path).exists()
    assert Path(html_path).exists()
