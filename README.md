# Spatial Audio Converter

A queue-backed Python service for converting ordinary audio into configurable, headphone-oriented binaural/spatial audio with explicit DSP stages and automated before/after quality analysis.

## Problem

A minimal “8D audio” effect can be built from opposing channel gains, but that is not the same as a binaural renderer. This project separates source motion, spatial filtering, room response, loudness control, quality measurement, encoding, and job orchestration so each stage can be tested and replaced independently.

## Algorithm

```text
input audio
    ↓
format decode + float normalization
    ↓
source analysis
    ↓
azimuth trajectory
    ↓
┌─────────────────────────────────────────┐
│ spatial renderer                        │
│  ├─ binaural HRIR/HRTF convolution       │
│  └─ explicit equal-power + ILD + ITD     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ room response                            │
│  ├─ early reflections                    │
│  ├─ Schroeder/Moorer late reverberation  │
│  └─ optional measured WAV impulse resp.  │
└─────────────────────────────────────────┘
    ↓
RMS normalization + peak limiting
    ↓
quality analysis
    ↓
WAV / MP3 encoding
    ↓
JSON + HTML quality reports
```

## Architecture

The web service uses the Phase-6 request boundary:

```text
HTTP API / UI
      ↓
JobManager
      ↓
bounded FIFO queue
      ↓
dedicated worker threads
      ↓
AudioPipeline
      ↓
LocalStorage
```

The HTTP request stages the uploaded bytes and enqueues a task; decode, DSP, analysis, encoding, and report generation happen in workers. The queue and job registry are intentionally in-process for this reference deployment.

## Phase 1 — correctness

The repository has automated tests for configuration bounds, file validation, API lifecycle, normalization, limiting, deterministic WAV/MP3 conversion, spatial-output shape/finiteness, and queue/worker lifecycle.

## Phase 2 — validated spatial baseline

The transparent fallback renderer uses equal-power panning, explicit ILD, and time-varying ITD.

```text
azimuth(t) = 90° · depth · sin(2π f t)

L = cos(θ / 2)
R = sin(θ / 2)
L² + R² = 1
```

## Phase 3 — actual binaural spatial audio

The primary spatial path is now convolutional: directional HRIRs are selected from a direction-indexed database and convolved into independent left/right ear signals.

### Synthetic HRIR fallback

The repository contains a deterministic synthetic FIR bank covering azimuths from `-90°` to `+90°`. It provides directional delay, near/far-ear attenuation, and short spectral structure so the code path is genuinely FIR-based even without an external dataset.

It is explicitly an engineering approximation, not a measured human HRTF.

### Measured SOFA input

A measured SOFA FreeFieldHRIR/SimpleFreeFieldHRIR dataset can be selected through:

```text
hrtf_source = sofa
hrtf_sofa_path = /path/to/listener.sofa
```

The loader expects two receiver channels (`Data.IR[M,2,N]`), `SourcePosition`, and `Data.SamplingRate`.

### Headphone-oriented rendering

Binaural mode produces the two ear signals intended for headphone playback. The legacy explicit stereo renderer remains available as a controlled fallback for comparison and debugging.

## Phase 4 — better room simulation

The former compact delay effect has been replaced by a selectable room stage.

### Early reflections

Multiple discrete reflections are generated with independent delays and small left/right gain differences. This makes the first-arrival structure explicit and inspectable.

### Schroeder/Moorer late field

The algorithmic late response uses:

```text
parallel feedback comb filters
        ↓
frequency damping
        ↓
series all-pass diffusion
        ↓
stereo wet field
```

Room size controls reflection/feedback time scales; damping controls late-field high-frequency loss.

### Measured room impulse response

A mono/stereo WAV impulse response can replace the algorithmic room model. The IR is resampled to the programme sample rate before convolution.

```text
room_model = measured-wav
room_ir_path = /path/to/room.wav
```

## Phase 5 — quality engineering

Each conversion produces a deterministic before/after quality report.

### Loudness and dynamics

The report includes:

```text
integrated LUFS
RMS dBFS
sample peak dBFS
4× oversampled true peak dBFS
crest factor dB
```

Integrated loudness is measured with the BS.1770-style meter exposed by `pyloudnorm`.

### Spectral / frequency-response analysis

The report stores a log-spaced magnitude profile plus spectral centroid, spectral rolloff, and band levels across:

```text
20–80 Hz
80–250 Hz
250–1000 Hz
1–4 kHz
4–10 kHz
10–20 kHz
```

### SNR and before/after comparison

The input/output mono downmixes are compared for:

```text
downmix SNR
spectral-response RMSE distance
centroid delta
rolloff delta
band-level deltas
```

These metrics are objective engineering diagnostics, not perceptual listening scores.

### Reports

Successful jobs produce:

```text
<output>.<ext>.quality.json
<output>.<ext>.quality.html
```

The JSON report is intended for automated regression/dashboards; the HTML report is intended for human inspection.

## Product/API integration

The current service exposes presets, asynchronous jobs, batch conversion, waveform/preview/download endpoints, and the Phase-3–5 configuration through the same queue-backed API.

Completed jobs expose audio plus machine-readable quality artifacts. The browser product can therefore visualize the same pipeline state and measurements rather than using a separate client-side model.

## Installation

Requirements:

- Python 3.11 or 3.12
- FFmpeg available on `PATH`

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate

pip install -e ".[test,tools]"
```

Run:

```bash
python wsgi.py
```

Open `http://localhost:5000`.

Quality checks:

```bash
ruff check .
pytest -q
```

## Benchmarks

The repository contains a deterministic synthetic-input pipeline benchmark:

```bash
python benchmarks/benchmark_pipeline.py --duration 10 --sample-rate 44100
```

The benchmark reports elapsed time and realtime factor. Performance values should only be compared when CPU, Python/NumPy/SciPy versions, FFmpeg build, input duration, and renderer configuration are controlled.

## Limitations

- The bundled synthetic HRIR bank is not a measured HRTF and is not a perceptual validation dataset.
- HRTF direction selection currently uses nearest-direction block selection; continuous direction interpolation and fractional-delay processing are future work.
- The SOFA loader targets two-receiver free-field HRIR conventions rather than every SOFA convention.
- The algorithmic room stage is a compact DSP model, not a physical acoustic field solver.
- Measured room support currently accepts WAV IRs rather than arbitrary spatial room formats.
- LUFS, SNR, and spectral metrics are objective measurements and do not establish perceived externalization or spaciousness.
- Job state and artifacts remain local/in-process and are not durable across process restarts.

## Future work

1. continuous/interpolated HRTF trajectories and fractional-delay ITD
2. measured listener-specific SOFA datasets
3. binaural room impulse responses and more physical room models
4. standards-aware loudness targeting with true-peak compliance
5. streaming/block processing for long-form audio
6. perceptual listening-test datasets and automated subjective evaluation
7. durable distributed queue/job persistence and object storage

## License

MIT. See `LICENSE`.
