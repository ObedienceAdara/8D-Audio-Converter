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
3D direction trajectory
    ↓
┌─────────────────────────────────────────┐
│ spatial renderer                        │
│  ├─ continuous HRTF/HRIR interpolation  │
│  │   ├─ bilinear grid interpolation     │
│  │   ├─ spherical K-neighbor weighting  │
│  │   └─ frequency-domain HRIR blending   │
│  ├─ trajectory smoothing                 │
│  ├─ directional filter crossfading       │
│  └─ equal-power + ILD + ITD fallback     │
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

The primary spatial path is convolutional: directional HRIRs are rendered into independent left/right ear signals.

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

## Phase 9 — continuous HRTF interpolation

The renderer no longer has to jump directly from one measured/synthetic HRIR to the nearest direction. A trajectory sample is mapped to a 3D unit direction and compared against neighboring HRTF measurements.

```text
source trajectory
      ↓
azimuth + elevation
      ↓
3D unit direction
      ↓
neighbor search on sphere
      ↓
┌──────────────────────────────┐
│ interpolation                │
│  ├─ nearest (baseline)       │
│  ├─ bilinear grid            │
│  └─ spherical weighted       │
└──────────────┬───────────────┘
               ↓
frequency-domain complex HRIR blend
               ↓
short causal FIR
               ↓
smooth filter-transition control
               ↓
binaural convolution
```

### Interpolation modes

`nearest` preserves the old directional lookup behavior and is useful as a baseline. `bilinear` uses the four corners when the SOFA measurements form a complete azimuth/elevation grid. `spherical` uses configurable K-neighbor inverse-distance weighting on the sphere and is the default for irregular measurement layouts.

Example configuration:

```text
hrtf_interpolation_quality = spherical
hrtf_interpolation_neighbors = 4
hrtf_filter_crossfade_blocks = 2
hrtf_trajectory_smoothing = 0.15
```

### Why frequency-domain interpolation is used

The selected HRIRs are transformed with a common real FFT, their complex spectra are combined using the directional weights, and the blended response is transformed back to a causal FIR. Because all selected responses use the same ear/tap representation, this avoids repeatedly designing a separate interpolation filter in the time domain and makes the interpolation rule explicit.

### Circular azimuth handling

Azimuth is normalized on `[-180°, 180°)`, and spherical distance is computed from 3D unit vectors rather than a raw linear degree difference. This means the `+179°` and `-179°` directions remain neighbors across the wrap seam.

### Trajectory smoothing

The azimuth path is unwrapped before smoothing so a seam crossing does not create an artificial `358°` jump. The smoothed trajectory is then normalized back onto the circular domain before HRTF lookup.

### Directional filter transitions

Successive interpolated filters can be crossfaded over configurable blocks. This adds a second layer of temporal continuity on top of the spatial interpolation itself, reducing abrupt filter changes when a rapidly moving trajectory crosses measurement regions.

### Three-dimensional API surface

`HRTFConvolver.process()` accepts an optional elevation trajectory in addition to azimuth. Existing horizontal 8D motion remains compatible, while measured SOFA grids with multiple elevations can now be interpolated in 3D.

### Important limitation

The current interpolator is a robust directional interpolation layer, not a full spherical-harmonic HRTF representation. It does not yet implement a physically exact HRTF field interpolation model, individualized HRTF fitting, or fractional-delay reconstruction.

## Product/API integration

The current service exposes presets, asynchronous jobs, batch conversion, waveform/preview/download endpoints, and the Phase-3–9 configuration through the same queue-backed API.

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
- HRTF interpolation is based on nearest/bilinear/spherical directional weighting and does not yet use spherical-harmonic reconstruction or fractional-delay interpolation.
- The SOFA loader targets two-receiver free-field HRIR conventions rather than every SOFA convention.
- The algorithmic room stage is a compact DSP model, not a physical acoustic field solver.
- Measured room support currently accepts WAV IRs rather than arbitrary spatial room formats.
- LUFS, SNR, and spectral metrics are objective measurements and do not establish perceived externalization or spaciousness.
- Job state and artifacts remain local/in-process and are not durable across process restarts.

## Future work

1. higher-order spherical/HRTF-field interpolation and fractional-delay ITD
2. partitioned overlap-save convolution for lower CPU cost on long recordings
3. listener-specific measured SOFA datasets and HRTF personalization
4. binaural room impulse responses and more physical room models
5. standards-aware loudness targeting with true-peak compliance
6. real-time streaming/head tracking and multi-source scenes
7. perceptual listening-test datasets and automated subjective evaluation
8. durable distributed queue/job persistence and object storage

## License

MIT. See `LICENSE`.
