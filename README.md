# Spatial Audio Converter

A modular Flask service for converting ordinary audio into a configurable spatial/"8D" listening effect.

## Current processing architecture

```text
Web UI / API
     |
     v
Job Manager
     |
     v
Audio Pipeline
     |
     +--> Format Decoder ----+
     +--> Signal Analysis ---+--> Spatial Engine --> Room / Reverb --> Loudness --> Quality Metrics --> Encoder --> Output Storage
     +--> Metadata -----------+        |
                                      +--> Azimuth Trajectory
                                      +--> Equal-Power Panning
                                      +--> ILD Model
                                      +--> ITD Model
                                      +--> Optional analytic HRTF
```

## Phase 1 — correctness

The project now has automated coverage for:

- processing-parameter bounds and defaults;
- supported/unsupported input formats and missing files;
- decoded-audio duration limits;
- API health, upload validation, status, and download contracts;
- bounded job execution and result persistence;
- RMS normalization, peak limiting, and silence behavior;
- WAV and MP3 end-to-end conversion when FFmpeg is available;
- spatial-output shape, finiteness, and frame-count invariants.

CI installs FFmpeg, runs Ruff, and runs the full Pytest suite.

## Phase 2 — spatial DSP baseline

### Equal-power panning

For azimuth `a` in degrees, where `-90°` is hard left, `0°` is center, and `+90°` is hard right:

```text
θ = radians(a + 90°)
L = cos(θ / 2)
R = sin(θ / 2)
```

The implementation enforces the constant-power invariant:

```text
L² + R² = 1
```

for every azimuth.

### Azimuth trajectory

The default trajectory is a smooth sinusoidal sweep:

```text
azimuth(t) = 90° × depth × sin(2π f t)
```

where `f` is the requested pan speed in hertz and `depth` is in `[0, 1]`.

### Interaural level difference (ILD)

The compact Phase-2 model attenuates the far ear as a function of azimuth magnitude:

```text
a = |sin(azimuth)|
attenuation_dB = max_far_ear_attenuation_dB × a
far_gain = 10^(-attenuation_dB / 20)
```

The near ear remains at unity in this model. Positive azimuth means the source is to the right; negative azimuth means it is to the left.

### Interaural time difference (ITD)

The Phase-2 ITD model uses a bounded maximum delay and converts it to integer sample offsets:

```text
delay_seconds = max_delay_seconds × |sin(azimuth)|
delay_samples = round(delay_seconds × sample_rate)
```

The delayed ear follows the source side and the delay is evaluated from the complete azimuth trajectory, rather than taking one maximum delay for the entire file.

## Important HRTF note

The optional HRTF component currently provides an analytic binaural approximation composed from the explicit ILD and ITD models. It is **not** a measured HRTF database and must not be treated as a perceptually validated production binaural renderer. The module boundary is intentionally replaceable so measured, subject-specific HRTFs can be introduced in a later phase.

## Supported input formats

MP3, WAV, FLAC, OGG, M4A and AAC, subject to FFmpeg support in the deployment environment. Uploads are capped at 32 MB and decoded audio is capped at 15 minutes by default.

## Local development

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -e ".[test,tools]"
```

FFmpeg must be installed and available on `PATH`.

Run the development server:

```bash
python wsgi.py
```

Then open `http://localhost:5000`.

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check .
```

## Docker

```bash
docker build -t spatial-audio-converter .
docker run --rm -p 5000:5000 spatial-audio-converter
```

The container runs one Gunicorn web worker with an internal bounded job pool. A multi-instance production deployment should replace the in-process queue with a shared worker backend.

## Repository layout

```text
.
├── src/spatial_audio_converter
│   ├── analysis/          # signal measurements
│   ├── api/               # Flask application and HTTP API
│   ├── audio/             # decode, encode, metadata
│   ├── domain/            # typed audio/pipeline models
│   ├── effects/           # room/early-reflection effects
│   ├── jobs/              # job lifecycle and worker execution
│   ├── pipeline/          # end-to-end orchestration
│   ├── processing/        # loudness and quality metrics
│   ├── spatial/           # trajectory, panning, ILD, ITD, HRTF
│   ├── storage/           # local artifact storage
│   └── web/               # templates and static UI
├── tests/                 # unit, API, lifecycle, and conversion tests
├── Dockerfile
├── pyproject.toml
├── wsgi.py
└── README.md
```

## Next phases

The clean interfaces are intended to support measured HRTF datasets, fractional-delay ITD, streaming/block processing, standards-based LUFS metering, stronger room modeling, persistent distributed job queues, object storage, batch conversion, and perceptual benchmarking.

## License

MIT. See `LICENSE`.
