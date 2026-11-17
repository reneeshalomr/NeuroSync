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

## Spectral features

`neurosync.features.spectral` computes a one-sided Welch PSD (density in V²/Hz) for each EEG channel, trapezoid-integrated absolute band power (V²), relative band power, and Shannon spectral entropy in nats plus a normalized [0, 1] value. Relative power uses the total PSD integral in the configured analysis range (default 0.5–45 Hz); the default contiguous band definitions therefore sum to approximately 1, subject to numerical integration. Spectral entropy uses normalized PSD-bin mass in that same range, not a differential-entropy estimate. Results also include channel-wise mean and population-standard-deviation summaries.

The default Welch estimate uses 2-second Hann segments, 50% overlap, constant detrending, and mean averaging. Channels already marked bad in MNE metadata are excluded; this code does not automatically diagnose bad channels. The returned PSD records frequency bins; band powers depend on segment length, sample rate, preprocessing, and chosen limits. Features computed after filtering describe the filtered signal and may be attenuated near filter cutoffs. Result metadata records feature settings, excluded channel annotations, and source preprocessing. These descriptive statistics do not identify emotions, intentions, collaboration state, or interpersonal synchrony.

```powershell
.\.venv\Scripts\python.exe examples\extract_synthetic_spectral_features.py
.\.venv\Scripts\python.exe -m pytest
```

## Cross-participant coherence

`neurosync.collaboration.synchrony` computes descriptive, matched-channel magnitude-squared coherence between two MNE recordings explicitly marked synthetic. It uses SciPy's Welch estimator with a Hann window, constant detrending, configurable segment duration and overlap, and reports channel-level coherence plus arithmetic mean summaries over configured frequency bands. Only EEG channels present and not marked bad in both recordings are included. The two inputs must have matching sampling rates and sample counts. Coherence is dimensionless; input EEG is expressed in volts. Defaults use two-second segments and the same delta through gamma band limits as the spectral feature module.

The estimate depends on window length, segment overlap, preprocessing, and the number of segments. Finite-sample estimates can be biased upward. This first implementation performs no surrogate/control inference, confidence interval estimation, or multiple-comparison correction; the band and across-channel summaries are descriptive. Shared injected components in synthetic signals are known construction choices, not evidence of interpersonal neural synchrony, causal coupling, or a collaboration state. The API rejects inputs without synthetic provenance.

```powershell
.\.venv\Scripts\python.exe examples\analyze_synthetic_coherence.py
.\.venv\Scripts\python.exe -m pytest
```

## Exploratory visualization and notebook

Reusable Matplotlib functions in `neurosync.visualization` plot synthetic time series in µV, spectral PSD with V²/Hz labels, and descriptive channel coherence. Each figure labels its synthetic provenance, relevant units, and cautious interpretation. Figures are returned to callers and are not automatically saved into the repository.

Install the optional Jupyter tooling with the project interpreter, then open the notebook from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,notebook]"
.\.venv\Scripts\jupyter.exe notebook notebooks\synthetic_collaboration_exploration.ipynb
```

The notebook calls the reusable generator, preprocessing, feature, coherence, and plot APIs. It uses a fixed seed and retains no generated dataset. Its cells are exploratory and do not perform inferential tests or validate a human collaboration-state hypothesis.

## Synthetic condition baseline

Install the optional ML extra and run the baseline example:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ml]"
.\.venv\Scripts\python.exe examples\evaluate_synthetic_baseline.py
```

The target is the generator's `HIGH_SHARED_COMPONENT` or `LOW_SHARED_COMPONENT` setting, not an annotated human collaboration state. Each seed produces a paired high/low example; both rows stay together in a `StratifiedGroupKFold` split. The workflow reports class counts, majority-class and standardized logistic regression baselines, accuracy, balanced accuracy, macro F1, confusion matrices, and a Wilson interval for the proportion of held-out seed groups where both paired rows are correct. Scaling is fit only within the training fold. The first of four deterministic folds is reported; the interval treats seed groups as the sampling unit and does not support human-population inference. No performance on this synthetic fixture supports real-world or human-state validity.
