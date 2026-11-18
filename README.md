# NeuroSync

NeuroSync is a research-oriented project exploring whether physiological and behavioral signals can help estimate human-robot collaboration states. Phase 1 provides a reproducible synthetic EEG pipeline; NeuroSync 2.0 Phase 2 is planned as seven development sessions spanning simulation, alignment, multimodal evaluation, adaptive assistance, and an interactive browser dashboard. Synthetic and simulated results are not human-subject evidence, clinical validation, or proof of real-robot safety.

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
.\.venv\Scripts\python.exe -m pip install -e ".[dev,notebook,ml]"
.\.venv\Scripts\jupyter.exe notebook notebooks\synthetic_collaboration_exploration.ipynb
```

The notebook calls the reusable generator, preprocessing, feature, coherence, plot, and grouped baseline APIs. It uses fixed seeds and retains no generated dataset. Its cells are exploratory and do not perform inferential tests or validate a human collaboration-state hypothesis.

## Synthetic condition baseline

Install the optional ML extra and run the baseline example:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ml]"
.\.venv\Scripts\python.exe examples\evaluate_synthetic_baseline.py
.\.venv\Scripts\python.exe examples\run_synthetic_research_workflow.py
```

The target is the generator's `HIGH_SHARED_COMPONENT` or `LOW_SHARED_COMPONENT` setting, not an annotated human collaboration state. Each seed produces a paired high/low example; both rows stay together in a `StratifiedGroupKFold` split. The workflow reports class counts, majority-class and standardized logistic regression baselines, accuracy, balanced accuracy, macro F1, confusion matrices, and a Wilson interval for the proportion of held-out seed groups where both paired rows are correct. Scaling is fit only within the training fold. The standalone baseline example reports the first of four deterministic folds; the integrated workflow uses all four folds for out-of-fold predictions. The interval treats seed groups as the sampling unit and does not support human-population inference. No performance on this synthetic fixture supports real-world or human-state validity.

The end-to-end demonstration runs signal generation, MNE conversion, spectral features, coherence, plots, and grouped cross-validation. It saves a JSON report and three PNG figures under `outputs/synthetic_research_demo/`, which is ignored by Git; no generated signals or datasets are serialized.

## Phase 2: simulated HRC research pipeline

The optional `neurosync.simulation` package provides a headless PyBullet DIRECT shared-workspace handover with two dynamic sphere proxies driven by bounded PD force controllers toward scripted reference paths. Sampled trajectories and velocities are read from the physics engine. Behavioral telemetry includes per-step speed, inter-agent distance, whole-run speed Pearson correlation, idle duration, and scripted synchronized/desynchronized task-state intervals. These events are simulation ground truth from the authored phase schedule, not observations of human behavior or inferred collaboration quality. This is a simple controlled physics task, not a robot arm, human motion model, or deployment evaluation. NPZ export keeps arrays numeric and provenance/events as JSON; filenames must include `simulated` and generated files remain under ignored paths.

Install the simulator extra and run its example with the project interpreter:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,sim]"
.\.venv\Scripts\python.exe examples\run_simulated_handover.py
```

On Windows, the current [PyPI PyBullet 3.2.7 distribution](https://pypi.org/project/pybullet/#files) resolves to a source build and requires Microsoft C++ Build Tools. The checked project environment does not include that toolchain, and WSL2 is not installed. The [GitHub Actions Linux run](https://github.com/reneeshalomr/NeuroSync/actions/runs/44161231064) on Python 3.11 succeeded: it installed the declared `sim` extra, passed the test and quality suite, and ran the simulation example and real-engine movement assertion. This verifies the dynamic PyBullet DIRECT path on Linux, not on native Windows. The agents are scripted dynamic proxies under a simple PD controller; the example does not validate robot arms, human motion, or real-world interaction.

### Event-driven EEG and alignment

`generate_event_modulated_synthetic_eeg` takes a simulated task result and maps its scripted synchronized/desynchronized intervals to sample-level states. It modulates the injected shared waveform by documented software scales (1.0 and 0.05); these numbers construct a test signal and are not estimates of a neural mechanism. `annotate_raw_with_task_events` and `epoch_task_events` map simulation seconds to MNE annotations, onset events, and task-state metadata. The default assumes a common clock; callers can provide an explicit offset, and out-of-recording intervals/windows are rejected. Tests include a Linux CI integration case that runs real PyBullet, creates synthetic EEG from those timestamps, and constructs MNE annotations/epochs. This is a reproducible software integration, not evidence that real EEG contains task-dependent synchrony.

### Unified EEG and behavior features

`neurosync.features.extract_unified_features` produces one row per half-open scripted task event from an event-driven synthetic EEG record and matching simulated task telemetry. Rows include participant-wise band power/entropy summaries, descriptive matched-channel coherence, agent speeds/path lengths, distance summaries, event labels/source, seeds, and session ID. EEG samples use volts; behavior uses meters and meters/second. The extractor validates sample rates, temporal coverage, telemetry shapes, finite values, event order, and window length. Event labels are authored simulation ground truth. For later model evaluation, all rows from a simulation seed/run must remain in the same group. Run a real PyBullet example with:

```powershell
.\.venv\Scripts\python.exe examples\extract_simulation_features.py
```

The example writes a synthetic/simulated labeled CSV and provenance JSON under ignored `outputs/simulation_features/`; it does not create repository-tracked data.

### Comparative group-aware models

`neurosync.modeling.evaluate_multimodal_comparison` evaluates EEG-only, behavior-only, and combined logistic regression models with the same `StratifiedGroupKFold` splits. Every row from the same session/seed stays in one held-out fold. Scaling is fit inside each training-fold pipeline. Reports include out-of-fold accuracy, balanced accuracy, macro F1, confusion matrices, class balance, and fold variability. The target is the scripted simulation task-state label; a relationship between the synthetic EEG construction and that label is intentional and does not imply real predictive validity. To run a reproducible real-PyBullet comparison across 12 independent simulation seeds:

```powershell
.\.venv\Scripts\python.exe examples\evaluate_multimodal_models.py
```

The example writes a JSON metrics report and labeled comparison figure under ignored `outputs/multimodal_evaluation/`. The figure includes modality metrics and out-of-fold confusion matrices, with the synthetic target provenance in its title.

### Adaptive assistance demonstration

`RuleBasedAssistancePolicy` increases the robot-proxy controller effort and reduces its scripted target delay when the task's ground-truth phase is desynchronized or the human proxy crosses a configurable hesitation-duration rule. Every decision records simulation time, input kinematics/state, reason, target-delay setting, gains, force limit, and assistance level. `evaluate_rule_based_assistance` runs matched unassisted and assisted simulations and reports descriptive distances and robot motion for both conditions. The policy deliberately reads scripted state labels for a transparent control demonstration; it is not an inference policy, safety controller, validated human-assistance method, or evidence that assistance improved human outcomes.

```powershell
.\.venv\Scripts\python.exe examples\evaluate_adaptive_assistance.py
```

The example saves both simulation telemetry files, a per-step policy decision CSV, a JSON comparison report, and a provenance-labeled figure under ignored `outputs/adaptive_assistance/`.

## NeuroSync 2.0 interactive research dashboard

An interactive browser-based Streamlit dashboard is a required Phase 2 deliverable for Session 6, with full-pipeline integration on Session 7. Planned views include experiment status and synthetic/simulation provenance; selectable simulation runs with both agent trajectories, distance, velocity, and scripted phase events; participant/channel-selectable EEG traces, spectral power, and coherence; group-aware EEG-only, behavior-only, and multimodal metrics/confusion matrices; and adaptive assistance decisions with their input features and task context. The dashboard is not implemented yet; the roadmap marks it as future work rather than a current capability.

The intended Windows launch is through the project interpreter, after the dashboard and report schema land:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard\app.py
```

The final setup must document the real PyBullet runtime path. The current machine lacks MSVC C++ Build Tools and WSL is not installed, so native Windows simulation has not run. PyBullet's PyPI files page lists platform-specific wheels and the Windows install attempted here selected a source distribution that could not build without MSVC. WSL2/Linux is under consideration as the supported simulation environment; no WSL-based workflow has been verified on this machine yet.
