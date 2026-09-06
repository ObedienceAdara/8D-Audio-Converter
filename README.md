# Spatial Audio Converter

A modular Flask service for converting ordinary audio into a configurable spatial/"8D" listening effect.

## Architecture

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
                                      +--> Trajectory Generator
                                      +--> Equal-Power Panning
                                      +--> HRTF Convolution
```

## Processing stages

1. **Format decoder** uses pydub/FFmpeg to decode supported audio into normalized float samples.
2. **Signal analysis** computes basic objective diagnostics such as RMS, peak, crest factor and stereo correlation.
3. **Metadata** extracts common source tags and processing metadata.
4. **Spatial engine** generates a smooth azimuth trajectory, provides an equal-power-panning fallback, and can run an analytic binaural HRTF approximation.
5. **Room/reverb** adds configurable multi-tap early reflections.
6. **Loudness control** normalizes RMS level and applies a configurable peak ceiling.
7. **Quality metrics** measures the resulting signal before encoding.
8. **Encoder** writes WAV or MP3 output. MP3 defaults to 320 kbps.
9. **Output storage** manages short-lived local artifacts.
10. **Job manager** runs conversions outside the HTTP request thread through a bounded worker pool.

## Important DSP note

The included HRTF component is an **analytic approximation** designed to keep the repository self-contained. It is not a measured HRTF database and should not be treated as a perceptually validated production binaural renderer. The project architecture leaves a clean boundary for replacing it with measured subject-specific HRTFs later.

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

The container installs FFmpeg and starts Gunicorn with one web worker plus internal job workers. Keeping one Gunicorn worker is intentional because the reference `JobManager` is in-process. A multi-instance production deployment should replace it with a shared queue/backend.

## Repository layout

```text
.
├── src/spatial_audio_converter
│   ├── analysis/          # signal measurements
│   ├── api/               # Flask application factory and HTTP API
│   ├── audio/             # decode, encode, metadata extraction
│   ├── domain/            # typed pipeline models
│   ├── effects/           # room/reverb processing
│   ├── jobs/              # job lifecycle and worker execution
│   ├── pipeline/          # orchestration
│   ├── processing/        # loudness and quality metrics
│   ├── spatial/           # trajectory, panning, HRTF, spatial engine
│   ├── storage/           # local artifact storage
│   └── web/               # templates and static UI
├── tests/                 # unit/API tests
├── Dockerfile
├── pyproject.toml
├── wsgi.py
└── README.md
```

## Future upgrades

The clean boundaries are intended to support measured HRTF datasets, streaming/block processing, real LUFS metering, stronger room modeling, persistent job queues, object storage, batch jobs, and perceptual benchmarking without coupling those concerns to Flask.

## License

MIT. See `LICENSE`.
