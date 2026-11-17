# Development Log

This log records dated project decisions, implementation work, checks, and limitations. Entries should separate completed work from plans and state evidence without implying scientific results that were not measured.

## 2026-11-17 — Project initialization and planning

- Inspected repository: branch `main`; remote `origin` points to `https://github.com/ReneeShalom/NeuroSync.git`; no commits exist yet. Git reports `.venv/` as untracked.
- Environment: `.venv` metadata specifies Python 3.11.9, but that environment currently contains only pip and setuptools. The active `python` command resolves to Anaconda Python 3.13.9 with a broad package set; this is not the project venv and is not the target runtime.
- In the active Anaconda interpreter, NumPy, SciPy, Pandas, scikit-learn, Matplotlib, Plotly, pytest, and Jupyter are importable. MNE, PyTorch, and PyBullet are not. This records availability only; no packages were installed and no compatibility was validated.
- Created the initial seven-day roadmap and project engineering/scientific integrity guidance.
- No pipeline implementation, experiment, model, or scientific result has been produced.
- Next: review the plan, then establish package metadata and a reproducible Python 3.11 development environment before implementation.

## 2026-11-17 — Synthetic EEG generator

- Implemented `src/neurosync/data/synthetic_eeg.py` with a typed result container, explicit high/low shared-component conditions, seeded generation, volts-based signals, channel/time metadata, and compressed NPZ export with JSON metadata that does not require pickle.
- The signal fixture combines randomized sinusoidal mixtures in the delta, theta, alpha, beta, and gamma ranges with independent white background noise and low-frequency background. The low condition uses a small residual injected shared component; all other component/noise draws remain participant-specific.
- Added a command-line inspection example and deterministic tests for shape, timing, seed behavior, conditions, finite values, invalid sampling rate, and NPZ serialization.
- Added minimal `pyproject.toml` metadata for Python 3.11, NumPy runtime dependency, pytest development extra, src-layout package discovery, and test configuration.
- Scientific limitation: this is a constructed simulation, not human EEG or a validated physiological model. Shared oscillations exist because the generator injects them; they cannot establish interpersonal synchrony, causal interaction, or collaboration state.
- Installed the declared NumPy and pytest dependencies into `.venv` and ran `.\\.venv\\Scripts\\python.exe -m pytest --basetemp .venv\\pytest-tmp` from the repository root: 8 tests passed. The first attempt exposed restricted access to the default Windows pytest temp directory; using an ignored temp directory inside `.venv` resolved it. The generator example and `compileall` also completed with the project interpreter. No generated dataset is stored in the repository.

## 2026-11-17 — Scientific and engineering review

- Exposed the exact injected shared-component waveform in volts and in NPZ output, so condition construction can be tested directly rather than inferred from noisy cross-participant correlation.
- Added a deterministic same-seed comparison showing the high-condition injected component is exactly 20 times the low-condition component (scale 1.0 versus 0.05). This tests generator construction only; it is not evidence of neural synchrony.
- Added explicit Nyquist metadata and NumPy/PCG64 provenance. Tests cover the maximum generated sinusoidal frequency at several supported sampling rates and verify signal amplitude scale in volts.
- Documented the independent/shared band amplitudes and modulation/background frequency ranges in generation metadata. Nyquist validation includes amplitude-modulation sidebands, not only the carrier frequency.
- Expanded reproducibility coverage to signals, time, settings, and metadata; expanded NPZ round-trip checks using `allow_pickle=False`; added invalid sampling frequency, duration, channel count, and condition cases.
- Enforced `synthetic` in saved filenames and documented the requirement. Updated README scientific and frequency-limit notes and the architecture path.
- Final review validation with `.venv\\Scripts\\python.exe` (Python 3.11.9): `-m pytest --basetemp .venv\\pytest-tmp` passed all 30 tests; `examples\\generate_synthetic_eeg.py` completed; `-m compileall -q src tests examples` completed. No NPZ dataset was saved in the repository.

## 2026-11-17 — Day 2 MNE preprocessing

- Added an MNE conversion boundary that accepts only `SyntheticEEGData`, validates channel/time/unit contracts, creates separate RawArray recordings for participants A and B, and embeds synthetic provenance plus MNE version in Raw metadata.
- Added an immutable preprocessing configuration and copy-returning offline zero-phase FIR filtering, optional explicitly configured notch filtering, and optional average reference. Filter/reference parameters and MNE version are retained in metadata.
- Added event-locked MNE epochs with MNE-format event validation, recording-boundary checks, explicit event mapping, and no implicit baseline correction.
- Added a runnable synthetic preprocessing example. Its event samples are illustrative and not behavioral annotations. Documented that zero-phase filtering is offline, can smear event transients, and that generic channels have no montage.
- Dependencies added to project configuration: `mne>=1.6,<2` at runtime and `ruff>=0.12,<1` in the development extra. Installed versions during this session: MNE 1.13.2, SciPy 1.17.1, Ruff 0.17.5.
- Validation with `.venv\\Scripts\\python.exe` (Python 3.11.9): full suite `-m pytest --basetemp .venv\\pytest-tmp` passed all 49 tests; both synthetic examples completed; `-m compileall -q src tests examples` completed; `-m ruff check src tests examples` passed; `-m ruff format --check src tests examples` confirmed 11 files already formatted.
- Environment: MNE 1.13.2, SciPy 1.17.1, NumPy 2.4.6, pytest 8.4.2, Ruff 0.17.5. No human recordings or generated datasets were added.
