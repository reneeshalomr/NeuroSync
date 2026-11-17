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

## 2026-11-17 — Day 3 spectral feature extraction

- Added SciPy as an explicit runtime dependency and implemented Welch PSD extraction for EEG channels from synthetic-labeled MNE recordings.
- Added trapezoid-integrated absolute band power in V² and relative band power defined against total power over a configurable analysis range. Defaults use contiguous delta/theta/alpha/beta/gamma bands from 0.5–45 Hz.
- Added Shannon entropy of normalized PSD-bin mass in nats and normalized to [0, 1], plus channel-level mean/population-standard-deviation summaries.
- Excluded channels already flagged as bad in MNE metadata from the summaries and recorded those exclusions; no automatic bad-channel classifier is included.
- Feature outputs retain Welch window, overlap, detrending, segment length, bin spacing, analysis/band limits, source condition/seed, preprocessing metadata, units, and NumPy/SciPy/MNE versions. Features reject non-synthetic provenance, empty EEG picks, invalid Nyquist support, zero-power input, and frequency resolutions too coarse for a requested band.
- Added analytic sinusoid tests for PSD and V² power, relative-power partition, entropy calculation, channel summaries, deterministic outputs, invalid conditions, and integration with Day 2 MNE objects; added an example that analyzes unfiltered synthetic Raw data to avoid inheriting the Day 2 example's illustrative 1–40 Hz filter.
- Scientific limitation: band power and entropy depend on sampling, windowing, analysis limits, and prior filtering. They are descriptive signal statistics and are not direct measurements of emotion, intention, collaboration state, or interpersonal synchrony.
- Dependencies added: `scipy>=1.14,<2` runtime.
- Validation with `.venv\\Scripts\\python.exe` (Python 3.11.9): full suite `-m pytest --basetemp .venv\\pytest-tmp` passed all 61 tests; all three examples completed; `-m compileall -q src tests examples` completed; `-m ruff check src tests examples` passed; `-m ruff format --check src tests examples` confirmed all 15 Python files formatted.
- Environment for this run: NumPy 2.4.6, SciPy 1.17.1, MNE 1.13.2, pytest 8.4.2, Ruff 0.17.5. No generated data were retained in the repository.

## 2026-11-17 — Day 4 descriptive synchrony analysis

- Added `neurosync.collaboration.synchrony` to compute channel-matched magnitude-squared coherence with SciPy's Welch estimator for two MNE recordings carrying explicit synthetic provenance. EEG channels marked bad in either recording are excluded; equal sample rates and lengths are required.
- Added configurable segment duration, overlap, and frequency bands; outputs include frequency-resolved coherence, per-channel band means, across-channel band means, source condition/seed, parameters, units, software version, and exclusion metadata.
- Added tests comparing the estimator against SciPy directly, checking an exact shared sinusoid fixture, summaries, bad-channel exclusion, repeatability, provenance and shape/configuration validation, and undefined zero-power handling. These tests validate constructed software behavior only.
- Added a runnable synthetic-only example and documented Hann windows, frequency resolution, finite-sample upward bias, lack of surrogate/control inference and multiplicity correction, and the distinction between statistical association and interpersonal synchrony.
- No dependency changes. No surrogate-based null inference was implemented; results are descriptive only.
- Validation with `.venv\\Scripts\\python.exe` (Python 3.11.9): full suite `-m pytest --basetemp .venv\\pytest-tmp` passed all 73 tests; all four examples completed; `-m compileall -q src tests examples` completed; `-m ruff check src tests examples` passed; `-m ruff format --check src tests examples` confirmed all 19 Python files formatted.
- No generated datasets or human recordings were retained. The final change set contains only source, tests, an example, and project documentation.

## 2026-11-17 — Day 5 exploratory visualization

- Added reusable Matplotlib visualizations for selected participant time series (µV), channel spectral PSD and relative band summaries (V²/Hz), and matched-channel coherence. Figure titles/captions identify synthetic data and avoid interpreting statistical association as proof of interpersonal synchrony.
- Added `notebooks/synthetic_collaboration_exploration.ipynb`, which uses a fixed seed to generate a synthetic pair, convert to MNE, extract spectral features, compute descriptive coherence, and call the reusable plotting functions. The notebook stores no outputs or datasets in Git.
- Added tests for figure structure, provenance and unit labels, cautious coherence language, and invalid channel/participant selections.
- Added `matplotlib>=3.8,<4` as a direct dependency and optional `jupyter>=1,<2` extra. Installed declared development/notebook extras into `.venv` using `.venv\\Scripts\\python.exe -m pip install -e ".[dev,notebook]"` to execute the notebook.
- Validation with Python 3.11.9: full suite passed all 77 tests; all four examples completed; all five notebook code cells executed successfully using nbclient; compileall completed; Ruff check passed; Ruff format check confirmed 23 Python files formatted. The Jupyter kernel emitted Windows Proactor/TCP warnings during local execution; execution completed successfully and the notebook contains no saved outputs.
- No generated data or real human recordings were added.

## 2026-11-17 — Day 6 synthetic baseline classification

- Added `neurosync.modeling.synthetic_baseline` to construct paired HIGH/LOW generator-setting rows from the same seed per session, with pooled channel relative spectral power/entropy and descriptive coherence features. The target is explicitly the generator setting, not a human collaboration-state annotation.
- Added majority-class and standardized logistic regression baselines. A `StratifiedGroupKFold` split keeps both condition examples for each paired seed together; `StandardScaler` is in the model pipeline and is fit on training rows only. Outputs report class balance, accuracy, balanced accuracy, macro F1, confusion matrices, per-class test support, and a Wilson interval for the proportion of held-out seed groups where both paired condition rows are correct. The interval uses groups as independent units; it is not a human-population estimate.
- Added tests for dataset balance/reproducibility, pair integrity, no group overlap, metrics, determinism, and invalid protocol inputs; added a runnable example and documented the split, target, and inference limits.
- Added optional `scikit-learn>=1.4,<2` ML extra and installed it with `.venv\\Scripts\\python.exe -m pip install -e ".[dev,ml]"`. Installed scikit-learn version: 1.9.1.
- Validation with `.venv\\Scripts\\python.exe` (Python 3.11.9): full suite passed all 87 tests; all five examples completed; all five notebook code cells executed; compileall completed; Ruff check passed; Ruff format check confirmed 28 Python files formatted. Headless plotting tests use Matplotlib Agg to avoid the unavailable Tk GUI toolkit in this environment.
- The fixed example generated a balanced 12/12 class set, trained on 9 paired seed groups, and held out 3 paired groups. On that single synthetic fold the majority baseline scored 0.50 row accuracy and logistic regression scored 1.00 row accuracy (6/6 rows); the Wilson interval for the latter's held-out exact-pair success was 0.439–1.000. These are reproducible software-fixture outputs with very small test-group count, not scientific evidence or estimates of human-state prediction.
- No human recordings or generated datasets were added.

## 2026-11-17 — Day 7 integration and final evaluation

- Added grouped all-fold cross-validation with out-of-fold predictions: every paired seed group is held out exactly once, the scaler is fit only within each training fold, and fold plus aggregate metrics are returned. Wilson uncertainty is calculated over paired seed groups rather than treating paired rows as independent.
- Expanded the exploratory notebook with the grouped generator-condition baseline and added `examples/run_synthetic_research_workflow.py` to run signal generation, MNE conversion, spectral features, coherence, plotting, and four-fold evaluation end to end.
- The integrated example writes three clearly synthetic PNG plots and a provenance/metrics JSON report under ignored `outputs/synthetic_research_demo/`. It does not save signals or generated datasets. An integration test verifies those outputs and synthetic-only metadata.
- Final validation with Python 3.11.9: full suite passed all 90 tests; all five examples completed; all six notebook code cells executed; compileall completed; Ruff check passed; Ruff format check confirmed 29 Python files formatted.
- The fixed integrated synthetic run used 16 paired seed groups (32 balanced rows). Out-of-fold logistic regression accuracy/balanced accuracy/macro F1 and paired-group exact accuracy were each 1.00; the majority dummy accuracy was 0.50. This measures separability of this controlled generator's condition setting only; it is not human EEG, a human collaboration-state result, or evidence of real-world validity.
- Jupyter emitted a Windows Proactor event-loop warning and local-kernel TCP warning during execution; all notebook cells completed successfully. Headless plotting tests explicitly use Agg because Tk is unavailable in the project Python distribution.
- No human recordings, generated signal datasets, secrets, or ignored outputs were staged. Day 7 is complete.

## 2026-11-17 — NeuroSync 2.0 strategy reset and runtime investigation

- Preserved all Phase 1 commits and the clean `origin/main` history through `3c0ba8ac9eec47042b0ed9fe944c8342b2815f1a`. Replaced no Phase 1 code or history. The confirmed Phase 2 delivery plan is seven new sessions; it now explicitly reserves Session 6 for the required interactive Streamlit dashboard and Session 7 for integration/CI.
- Inspected the PyPI PyBullet install path in the Python 3.11.9 Windows `.venv`. `pip install -e ".[dev,sim]"` selected the 3.2.7 source archive and failed while building because Microsoft Visual C++ 14.0 or newer is not installed. `wsl.exe --list --quiet` reports WSL is not installed. No actual PyBullet simulation has run, and the real-engine smoke test remains skipped; no mock run is represented as real physics validation.
- Preserved uncommitted Session 1 simulation scaffolding and Session 2 alignment work in the working tree. The simulation contract tests use a test double and passed 4 tests, with the actual-engine smoke test skipped; the alignment suite passed 5 tests. These are partial checks, not completed milestones.
- Updated README and ROADMAP with the seven-session plan, the current runtime limitation, and the required dashboard scope and launch command (planned, not yet implemented).
- No PyBullet dependency install succeeded; no system tools or WSL were installed. No Phase 2 feature is marked complete in this entry. Next priority remains establishing a real PyBullet runtime without extending setup troubleshooting, then finishing and validating Session 1. Continue independent pipeline work if that runtime remains blocked.
- No Phase 2 commit or push was made in this strategy-reset work; the existing uncommitted feature work is intentionally preserved for continuation.

## 2026-11-17 — Phase 2 independent event pipeline work (Session 1 runtime still blocked)

- Continued independent work without describing the test double as PyBullet. The simulation module and run example are preserved and use PyBullet DIRECT when that optional engine is available. Contract tests use a test double; the actual-engine smoke test is explicitly skipped when the optional engine cannot import.
- Added simulation-backed task-event timestamps to MNE Raw annotations and task-onset Epochs. Alignment validates synthetic EEG provenance, simulated event provenance, clock offsets, interval coverage, event mapping, and epoch bounds; Epochs metadata includes phase/state/event source.
- Extended synthetic EEG records with optional sample task-state/event/simulation metadata and added an event-driven generator that removes the base low shared residual before injecting a controlled envelope (scale 1.0 in scripted synchronized phases, 0.05 in desynchronized phases). The parameters are software construction settings, not neurophysiological estimates. MNE Raw and pickle-free NPZ provenance retain simulation linkage.
- Added the explicit Pandas runtime dependency for MNE Epochs metadata. PyBullet remains optional under `[sim]`; its install did not succeed in this Windows environment because no MSVC C++ build tools are present.
- Focused checks passed: 5 alignment tests and 5 event-driven EEG tests. Simulation contracts passed 5 tests; the real PyBullet smoke test was skipped, and one optional-dependency test passed. Full project suite: 105 passed, 1 skipped. Six non-PyBullet examples ran. Compileall and Ruff check passed; all 38 Python files passed Ruff format check.
- Dashboard requirements are now documented for Session 6 but no dashboard code exists yet. The master pipeline and CI are also future work.
- Session 1 is not complete; Session 2's implementation is independently tested but its actual simulation integration remains unverified. Commit and push status for this delivery increment is reported in the session report.

## 2026-11-17 — Linux real-engine CI route

- Added `.github/workflows/ci.yml` on Ubuntu/Python 3.11 to install the project with `dev`, `ml`, and `sim` extras, run pytest, Ruff lint/format checks, compileall, and execute `examples/run_simulated_handover.py`. The final step requires the installed PyBullet engine and is intended to exercise the actual PyBullet DIRECT backend rather than a test double.
- The Windows `.venv` remains Python 3.11.9 but has no MSVC build tools; WSL2 is also unavailable. No local real-engine test was possible. Session 1 remains incomplete until the remote workflow succeeds.
- Validation on Windows Python 3.11.9: pytest reported 105 passed and 1 skipped (the conditional actual-PyBullet test). Ruff check passed, Ruff format check passed (38 files), and compileall passed. Pytest emitted a cache permission warning because this sandbox cannot write the repository `.pytest_cache`; it did not affect results.
- The remote `origin/main` at the start of this change was `319d85a9406a7576737083a28f6658a305f6db9e`. Commit `012192b8dc2138471a00cb8acce63d3e0046240f` was pushed and verified at `origin/main`. GitHub Actions run `44159751019` succeeded, including install, all tests, lint, formatting, compile, and the real PyBullet smoke example. This verifies the PyBullet DIRECT runtime on Linux; it does not establish native Windows support or validate dynamic/contact behavior because the agents are scripted kinematic proxies.
- Follow-up review found the initial proxies had zero mass and were moved by pose reset, so although the smoke run used the actual PyBullet package, it did not exercise dynamic agent motion. Replaced them with dynamic 0.25 kg spheres and bounded PD force control; sampled position and velocity telemetry now comes from PyBullet after each fixed step. Controller parameters and reference-path limitations are recorded in metadata. Updated the contract test double and added an actual-engine assertion that bodies move.
- Focused Windows Python 3.11.9 simulator tests: 5 passed, 1 skipped (PyBullet unavailable locally); Ruff check and format checks passed. The dynamic controller change has not yet been verified in Linux CI and requires a successful follow-up run before considering the physics change validated.
