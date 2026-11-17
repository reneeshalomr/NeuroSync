# NeuroSync

NeuroSync is a research-oriented project exploring whether physiological and behavioral signals can help estimate human-robot collaboration states. Initial work will focus on a reproducible pipeline using clearly labeled synthetic EEG data. Synthetic results are not human-subject evidence or clinical validation.

## Windows Python environment

The project targets Python 3.11.9. Use the repository virtual environment interpreter explicitly for every Python and pip command; do not rely on `python` resolution through `PATH`.

```powershell
# Confirm the project interpreter
.\.venv\Scripts\python.exe --version

# Run project scripts and tests
.\.venv\Scripts\python.exe path\to\script.py
.\.venv\Scripts\python.exe -m pytest

# Install declared project dependencies when setup is authorized
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The expected interpreter version is Python 3.11.9. If `.venv` must be recreated, create it with a Python 3.11.9 executable, then continue to invoke it explicitly as shown above. Do not use Anaconda Python 3.13 for this project.

See [ROADMAP.md](ROADMAP.md) for the staged plan and [AGENTS.md](AGENTS.md) for engineering and scientific integrity requirements.

## Synthetic EEG generator

The initial signal fixture is available at `src/neurosync/data/synthetic_eeg.py`. It generates two simulated participants with configurable sampling frequency, duration, channel count, seed, and one of two controlled shared-component conditions. Signals use volts (V), have shape `(channels, samples)`, and include delta, theta, alpha, beta, and gamma mixtures, low-frequency background, and independent broadband noise.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe examples\generate_synthetic_eeg.py
.\.venv\Scripts\python.exe examples\generate_synthetic_eeg.py --output data\generated\example_synthetic_eeg.npz
.\.venv\Scripts\python.exe -m pytest
```

NPZ files contain numeric arrays and JSON metadata and can be opened with `numpy.load(..., allow_pickle=False)`. Generated files under `data/generated/` are ignored by Git. The generator is a transparent software fixture, not a biophysical model or validated representation of real human EEG. Its injected shared oscillations do not prove interpersonal neural synchrony, causal interaction, or collaboration state. Synthetic tests verify software behavior only.

The saved file includes both participant signals, the injected shared-component waveform, the time vector, channel names, condition, seed, units, generator settings, and NumPy version. `save_synthetic_eeg` requires `synthetic` in the output filename to keep provenance visible. Carrier frequencies are at or below 45 Hz; the amplitude modulation can add sidebands up to 45.35 Hz. The generator therefore requires a sampling frequency above 90.7 Hz so those oscillatory components remain below Nyquist. Broadband discrete noise is sampled on the same time grid and is necessarily limited by its Nyquist frequency.

## MNE preprocessing

The Day 2 preprocessing package converts each synthetic participant to a separate MNE `RawArray`, validates the signal/time/channel contract, applies an explicit offline zero-phase FIR filter, and can apply an average reference. Notch filtering is opt-in because a mains frequency is not assumed. Event-locked epochs require explicit MNE-format event samples and event IDs; baseline correction defaults to `None`. Preprocessing settings and synthetic provenance are retained in MNE metadata.

```powershell
.\.venv\Scripts\python.exe examples\preprocess_synthetic_eeg.py
.\.venv\Scripts\python.exe -m pytest
```

Filtering is offline and can smear transients around events; these settings are not an online robot-control pipeline. The example's event indices are illustrative and are not task annotations. No montage is assigned to the generic synthetic channel names. Filter/reference choices are workflow defaults and require task- and device-specific justification for real recordings.
