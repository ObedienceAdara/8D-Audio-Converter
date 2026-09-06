# Spatial Audio Converter

A queue-backed Python service for turning ordinary audio into a configurable spatial/"8D" listening effect.

The project is intentionally structured like an engineering system rather than a single DSP script: the HTTP layer accepts work, a bounded queue decouples requests from processing, dedicated workers run the audio pipeline, and storage owns short-lived artifacts.

## Problem

Basic "8D audio" implementations often reduce spatialization to a pair of opposing amplitude multipliers. That is easy to implement, but it mixes motion, gain control, and channel routing into one effect and makes correctness difficult to test.

This project separates those concerns. A source azimuth trajectory is generated explicitly, equal-power panning establishes the stereo baseline, interaural level difference (ILD) and interaural time difference (ITD) are modeled independently, and downstream room, loudness, quality, and encoding stages operate on explicit intermediate representations.

## Algorithm

The validated Phase-2 baseline is:

```text
input audio
    ↓
format decode + float normalization
    ↓
azimuth trajectory
    ↓
equal-power panning
    ↓
interaural level difference (ILD)
    ↓
interaural time difference (ITD)
    ↓
optional early-room reflections
    ↓
RMS normalization + peak limiting
    ↓
quality metrics
    ↓
WAV / MP3 encoding
```

The default spatial renderer is explicit equal-power + ILD + ITD. The repository also contains an analytic HRTF approximation, but it is opt-in and is not presented as a measured HRTF dataset.

## Math

### Azimuth trajectory

For a configured movement frequency `f` and depth `d`:

```text
azimuth(t) = 90° · d · sin(2πft)
```

The trajectory is bounded by the configured depth and is generated independently from the panning stage.

### Equal-power panning

For a normalized azimuth mapped to an equal-power pan angle `θ`:

```text
L = cos(θ / 2)
R = sin(θ / 2)
```

The implementation is tested against the energy invariant:

```text
L² + R² = 1
```

so the spatial move does not use the legacy linear `1 ± p` amplitude modulation.

### Interaural level difference

The simplified ILD stage attenuates the far ear as a function of lateral source position. The current model uses an explicit bounded dB attenuation envelope rather than an implicit channel multiplier hidden inside the panner.

### Interaural time difference

The simplified ITD stage derives an ear-specific delay from the instantaneous azimuth trajectory. Unlike a single delay chosen from the maximum excursion, the implementation follows the time-varying trajectory so motion produces time-varying arrival differences.

These are engineering models, not a claim of psychoacoustic equivalence to a measured head-related transfer function.

## Architecture

```text
                         ┌──────────────────┐
                         │   Flask API/UI   │
                         └────────┬─────────┘
                                  │ submit / status / preview
                                  ▼
                         ┌──────────────────┐
                         │    JobManager    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Bounded FIFO     │
                         │ conversion queue │
                         └────────┬─────────┘
                                  │
                         ┌────────▼────────┐
                         │  worker threads │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  AudioPipeline   │
                         │ decode → DSP →   │
                         │ room → loudness  │
                         │ → metrics → enc. │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   LocalStorage   │
                         │ uploads/outputs  │
                         └──────────────────┘
```

The key architectural property is the request boundary:

```text
HTTP request
   │
   ├── validate metadata + parameters
   ├── stage uploaded bytes
   └── enqueue conversion task
             │
             └── HTTP returns 202

queue
   ↓
worker
   ↓
conversion pipeline
   ↓
storage
```

The upload copy still occurs inside the HTTP request because the server must receive the file before it can queue processing. Audio decoding, DSP, analysis, encoding, and artifact production do not block the request thread.

The reference queue is in-process. A production multi-instance deployment should replace it with a durable shared queue/backend.

## Product

The web UI now includes:

- waveform rendering for the selected source and completed output
- browser-side source preview and server-backed output preview
- Balanced, Deep, and Clean presets
- a live visual azimuth trajectory
- explicit processing progress and current pipeline stage
- single-file and batch conversion
- output format selection (WAV / MP3) and MP3 bitrate selection
- post-processing metrics and downloadable artifacts

## API

### `GET /health`

Returns service health, queue depth, and active worker count.

### `GET /api/presets`

Returns the built-in product presets and their parameter values.

### `POST /api/jobs`

Accepts one audio file and returns `202 Accepted` with a job identifier and status URL.

### `GET /api/jobs/<job_id>`

Returns job lifecycle state, progress, current stage, metrics, and—when complete—download, preview, and waveform URLs.

### `GET /api/jobs/<job_id>/preview`

Serves a completed output inline for browser audio playback.

### `GET /api/jobs/<job_id>/download`

Serves a completed output as a download.

### `GET /api/jobs/<job_id>/waveform`

Returns the compact peak-envelope representation stored by the worker.

### `POST /api/batches`

Queues multiple uploaded files using one processing configuration and returns a batch status URL.

### `GET /api/batches/<batch_id>`

Returns aggregate batch progress and per-file job status.

## Benchmarks

The repository includes a deterministic pipeline benchmark:

```bash
python benchmarks/benchmark_pipeline.py --duration 10 --sample-rate 44100
```

It reports elapsed time and a realtime factor for the same fixed synthetic input. Performance numbers are intentionally not hard-coded into this README because meaningful comparisons require the same CPU, Python/NumPy/SciPy versions, FFmpeg build, input duration, and renderer configuration.

## Screenshots

The application is designed around four presentation views:

1. **Conversion workspace** — file selection, presets, DSP parameters, and format controls.
2. **Waveform + trajectory** — source envelope plus live azimuth visualization.
3. **Processing state** — queue depth, progress percentage, and pipeline stage.
4. **Rendered result** — output preview, output waveform, metrics, and download.

For a release-quality project page, capture these views at desktop and mobile widths and place them under `docs/screenshots/`. The README deliberately does not embed fabricated screenshots.

## Demo

Run the service locally:

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate

pip install -e ".[test,tools]"
python wsgi.py
```

Then open:

```text
http://localhost:5000
```

The benchmark command and test suite are also local, deterministic entry points for demonstrating the DSP and queue architecture.

## Installation

### Requirements

- Python 3.11 or 3.12
- FFmpeg available on `PATH`
- a normal local filesystem for short-lived upload/output artifacts

Install the project:

```bash
pip install -e ".[test,tools]"
```

Run quality checks:

```bash
ruff check .
pytest -q
```

Docker is also supported:

```bash
docker build -t spatial-audio-converter .
docker run --rm -p 5000:5000 spatial-audio-converter
```

The container installs FFmpeg and runs Gunicorn with one web worker. The in-process job manager owns the internal worker threads, which is deliberate for this reference deployment.

## Results

The current engineering result is a modular, testable spatial-audio service rather than a single-file effect script.

The repository now has explicit boundaries for:

```text
analysis/
audio/
domain/
effects/
jobs/
pipeline/
processing/
spatial/
storage/
api/
web/
tests/
benchmarks/
```

The correctness suite covers configuration, file validation, API contracts, queue/worker lifecycle, normalization, waveform generation, conversion, and spatial DSP invariants. CI runs linting and the test suite with FFmpeg installed.

## Limitations

- The queue and job registry are in-process and are not durable across process restarts.
- Local artifact storage is not object storage and is not designed for multi-node deployments.
- ITD is currently represented with an integer-sample delay model; fractional-delay filtering is a future DSP improvement.
- ILD is a simplified bounded model, not a measured frequency-dependent binaural response.
- The optional HRTF implementation is analytic rather than a measured HRTF database.
- The room stage is a compact early-reflection model rather than a full acoustic simulator.
- The loudness stage is RMS-oriented; it is not a full standards-based LUFS implementation.
- The project currently optimizes for a transparent, inspectable reference architecture rather than cloud-scale job persistence.

## Future work

The next high-value extensions are:

1. durable queues and persistent job metadata
2. object storage and cleanup policies for multi-instance deployments
3. fractional-delay ITD and more physical ear geometry
4. measured HRTF datasets with selectable listener profiles
5. frequency-dependent ILD and spectral coloration
6. stronger room/reverberation modeling
7. standards-based loudness metering
8. streaming/block-wise processing for long-form audio
9. automated perceptual evaluation and listening-test datasets
10. richer benchmark reporting across CPUs, sample rates, durations, and renderers

## License

MIT. See `LICENSE`.
