# Spatial Audio Converter

A modular Flask service for converting ordinary audio into headphone-oriented binaural/spatial audio with explicit DSP stages, configurable room simulation, and automated before/after quality analysis.

## Processing architecture

```text
Web API / UI
    |
    v
Job Manager
    |
    v
Audio Pipeline
    |
    +--> Decode + normalize
    +--> Source analysis
    +--> Spatial renderer
    |      +--> Equal-power + ILD + ITD fallback
    |      +--> HRTF/HRIR binaural convolution
    +--> Room simulation
    |      +--> Early reflections
    |      +--> Schroeder/Moorer late field
    |      +--> Optional measured WAV IR
    +--> RMS / peak limiting
    +--> Quality analysis
    |      +--> LUFS
    |      +--> RMS / peak / true peak
    |      +--> crest factor
    |      +--> spectral profile / frequency response
    |      +--> SNR / spectral distance
    +--> WAV / MP3 encode
    +--> JSON + HTML quality report
```

## Phase 1 — correctness

The project has automated coverage for configuration bounds, input validation, decoded-audio limits, API lifecycle, normalization, limiting, output encoding, spatial-output shape/finiteness, and deterministic conversion fixtures.

CI installs FFmpeg, runs Ruff, and runs the Pytest suite.

## Phase 2 — validated spatial baseline

The fallback renderer remains explicit equal-power panning plus ILD and time-varying ITD.

For azimuth `a`:

```text
θ = radians(a + 90°)
L = cos(θ / 2)
R = sin(θ / 2)
L² + R² = 1
```

The motion trajectory is:

```text
azimuth(t) = 90° · depth · sin(2π f t)
```

This is the transparent reference path used when binaural HRTF rendering is disabled.

## Phase 3 — actual binaural spatial audio

### HRTF / HRIR convolution

The primary binaural path now works with head-related impulse responses rather than only amplitude/pan formulas.

The renderer operates on a mono source and maps each time block to the nearest available source direction. The selected left/right HRIRs are convolved with the block and overlap-added into a two-channel output.

```text
mono source
    |
    +--> azimuth trajectory
    |
    +--> directional HRIR selection
    |       +--> left ear FIR
    |       +--> right ear FIR
    |
    +--> convolution
    |
    v
left/right headphone signal
```

### HRTF sources

The repository supports two explicit sources:

1. **Synthetic HRIR bank** — deterministic, self-contained FIR filters used for development and tests. This is an engineering approximation, not a measured human HRTF database.
2. **SOFA file** — measured HRTF/HRIR data can be loaded from a `SimpleFreeFieldHRIR`/`FreeFieldHRIR`-style SOFA file containing `Data.IR`, `SourcePosition`, and `Data.SamplingRate`.

Configure a measured dataset through:

```text
hrtf_source = sofa
hrtf_sofa_path = /path/to/listener.sofa
```

or through `AUDIO_HRTF_SOFA_PATH` for the low-level renderer.

### Headphone orientation

Binaural rendering is the default product mode because the two output channels represent the left/right ear signals intended for headphone playback. The explicit stereo renderer remains available as a transparent fallback.

## Phase 4 — room simulation

The old single delay/tap effect has been replaced by a selectable room stage.

### Early reflections

The algorithmic room model creates several discrete reflections with distinct delays and slightly different left/right gains. This makes the first-order spatial response inspectable rather than hiding everything inside one delay.

### Schroeder/Moorer-style late field

The late field is assembled from:

```text
parallel feedback comb filters
          ↓
frequency damping
          ↓
series all-pass diffusion
          ↓
stereo wet field
```

Room size changes the reflection and feedback time scales. Damping controls the late-field low-pass behavior.

### Measured room impulse response

A mono or stereo WAV room impulse response can replace the algorithmic model. The IR is resampled to the programme sample rate and normalized before convolution.

Select:

```text
room_model = measured-wav
room_ir_path = /path/to/room.wav
```

## Phase 5 — quality engineering

Every completed conversion produces a machine-readable before/after quality report.

### Loudness

Integrated loudness is measured with `pyloudnorm` using its BS.1770-style meter configuration. The implementation exposes the resulting LUFS value rather than pretending RMS and LUFS are interchangeable.

### Dynamics

The report includes:

```text
RMS dBFS
sample peak dBFS
4× oversampled true peak dBFS
crest factor dB
```

### Spectral comparison

A deterministic log-spaced spectrum profile is recorded for the input and rendered output, together with:

```text
spectral centroid
spectral rolloff
frequency-response profile
20–80 Hz level
80–250 Hz level
250–1000 Hz level
1–4 kHz level
4–10 kHz level
10–20 kHz level
```

### SNR / distance

The report compares the mono downmix of the source and rendered result and records:

```text
downmix SNR
spectral-response RMSE distance
band-level deltas
centroid delta
rolloff delta
```

These are objective engineering measurements, not perceptual quality scores.

### Automated artifacts

For each successful output, the pipeline writes:

```text
<output>.<ext>.quality.json
<output>.<ext>.quality.html
```

The JSON is intended for automated regression testing and dashboards. The HTML report is intended for engineers reviewing an individual conversion.

## API

### `POST /api/jobs`

Submit one audio file and receive the asynchronous job identifier.

### `GET /api/jobs/<job_id>`

Returns status, waveform, metrics, and completed artifact URLs.

### `GET /api/jobs/<job_id>/download`

Downloads the rendered audio.

### `GET /api/jobs/<job_id>/preview`

Streams the completed output for browser playback.

### `GET /api/jobs/<job_id>/waveform`

Returns the compact output waveform representation.

### `GET /api/jobs/<job_id>/report`

Downloads the machine-readable JSON quality report.

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
pytest -q
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

## Repository layout

```text
.
├── src/spatial_audio_converter
│   ├── analysis/          # signal and waveform analysis
│   ├── api/               # Flask HTTP surface
│   ├── audio/             # decode, encode, metadata
│   ├── domain/            # typed audio/pipeline models
│   ├── effects/           # room simulation
│   ├── jobs/              # job lifecycle / worker boundary
│   ├── pipeline/          # end-to-end orchestration
│   ├── processing/        # loudness, QA, reports
│   ├── spatial/           # trajectory, panning, ILD, ITD, HRTF
│   ├── storage/           # local artifact storage
│   └── web/               # HTML/CSS/JS UI
├── tests/                 # DSP, API, lifecycle, and QA tests
├── Dockerfile
├── pyproject.toml
├── wsgi.py
└── README.md
```

## Limitations

- The bundled synthetic HRIR bank is not a measured human HRTF and should not be presented as psychoacoustically validated.
- Direction switching currently selects the nearest HRIR per processing block; fractional-delay interpolation and smoother continuous HRTF interpolation are future improvements.
- The SOFA path currently targets two-receiver free-field HRIR data rather than every possible SOFA convention.
- The room simulation is a compact engineering model, not a physical acoustic field solver.
- The measured-room path currently accepts WAV impulse responses rather than arbitrary spatial room formats.
- The loudness measurements are objective diagnostics; they do not establish perceived spaciousness or headphone externalization.
- The SNR comparison intentionally uses a downmixed source/reference pair. It is not a universal perceptual score.
- The current artifact storage and job manager remain local/in-process and are not a durable distributed processing system.

## Future work

1. continuous/interpolated HRTF trajectories and fractional-delay ITD
2. binaural room impulse responses and true direction-dependent room responses
3. measured SOFA listener profiles with selectable datasets
4. stronger perceptual and listening-test evaluation
5. LUFS-based target loudness control with true-peak compliance
6. streaming/block-wise processing for long files
7. richer benchmark matrices across hardware, sample rate, room size, and HRTF source
8. persistent distributed jobs and object storage

## Engineering references

- ITU-R BS.1770 family — objective programme loudness / true-peak measurement basis.
- EBU R 128 — loudness normalization and maximum true-peak guidance.
- SOFA conventions — standardized exchange of spatial audio / HRTF data, including FreeFieldHRIR-style measured impulse responses.
- `pyloudnorm` — implementation used for the programme loudness meter.

## License

MIT. See `LICENSE`.
