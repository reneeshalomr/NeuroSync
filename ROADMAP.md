# NeuroSync: Seven-Day Initial Roadmap

## Objective

Establish a reproducible, research-oriented foundation for studying whether physiological and behavioral signals can estimate human-robot collaboration states. The first milestone is a transparent synthetic EEG pipeline, not a claim about human participants, clinical use, or robot capability.

## Day 1 — Synthetic EEG generation (2026-11-17)

- Create reproducible, two-participant synthetic signals with controlled shared-component conditions.
- Record units, conditions, seed, settings, and simulation limitations; keep generated datasets out of Git.

**Exit criteria:** synthetic signals and metadata are reproducible and tested; no real-human or synchrony claim is made.

**Status (2026-11-17): complete.** The generator, NPZ provenance, deterministic condition test, and synthetic-only example are implemented and tested.

## Day 2 — EEG preprocessing with MNE

- Convert participants separately to MNE Raw objects with channel, sampling, unit, and synthetic provenance checks.
- Implement explicit offline filtering, optional notch filtering, configurable average reference, and event-locked epoching.
- Preserve preprocessing settings and require explicit event definitions and baseline choices.

**Exit criteria:** synthetic inputs pass documented MNE preprocessing and epoching checks; filtering and reference assumptions are documented and tested.

**Status (2026-11-17): implementation and validation complete.** The MNE converter, filtering/reference API, epoching API, tests, and walkthrough are present.

## Day 3 — Spectral feature extraction

- Define frequency bands, PSD method, windowing, units, and quality checks.
- Compare features against deterministic signal fixtures and document resolution and leakage considerations.

**Exit criteria:** feature definitions and tests are explicit; features are not described as direct measures of emotion or intention.

**Status (2026-11-17): implementation and validation complete.** Welch PSD, absolute/relative band powers, spectral entropy, channel summaries, tests, and a synthetic-only example are implemented. Values are descriptive and depend on the configured window, bands, and source preprocessing.

## Day 4 — Synchrony analysis

- Add descriptive inter-signal statistics with windowing, preprocessing, and surrogate/control assumptions stated.
- Test calculations on synthetic signals without treating constructed shared components as evidence about humans.

**Exit criteria:** outputs are described as statistical association; no coherence or phase metric is called proof of interpersonal synchrony.

**Status (2026-11-17): implementation and validation complete.** Added Welch magnitude-squared coherence for matched, non-bad EEG channels in synthetic MNE recordings, channel/band summaries, parameter and provenance metadata, tests against SciPy, and a runnable synthetic-only example. This milestone performs no surrogate/control inference or multiple-comparison correction and makes no human synchrony claim.

## Day 5 — Visualization and exploratory research notebook

- Build reusable plots and a notebook calling package functions for synthetic signals, features, and synchrony summaries.
- Label simulated data, units, parameters, and limitations in every figure.

**Exit criteria:** the notebook runs reproducibly and separates exploration from confirmatory evaluation.

**Status (2026-11-17): implementation and validation complete.** Added reusable signal, spectral, and coherence plots with synthetic provenance, units, and cautious captions; an exploratory notebook with a fixed seed that executes the complete synthetic pipeline; tests for plotting contracts; and an optional Jupyter dependency group. The notebook is descriptive and performs no inferential evaluation.

## Day 6 — Baseline ML for synthetic collaboration-state classification

- Define a clearly synthetic target and simple baseline models; use participant/session-aware splits where the data contract supports them.
- Report class balance, leakage controls, uncertainty, and evaluation limits.

**Exit criteria:** tests and metrics demonstrate software workflow only; no real-world validity claim is made.

**Status (2026-11-17): implementation and validation complete.** Added balanced paired synthetic-condition rows with shared-seed groups, spectral/coherence feature assembly, a majority dummy and train-fold-scaled logistic baseline, stratified group holdout, class counts, confusion and summary metrics, and a Wilson interval over held-out paired seed groups. The target is the synthetic generator setting, not a human collaboration-state label.

## Day 7 — Integration, evaluation, documentation, and demonstration

- Run the end-to-end synthetic workflow, full test suite, compilation, and code-quality checks.
- Review provenance, generated-file exclusions, scientific language, and installation instructions.
- Summarize completed work, measured checks, remaining risks, and future research tasks.

**Exit criteria:** the staged synthetic platform is reproducible and reviewable; no real participant data or robot deployment claim is implied.

**Status (2026-11-17): implementation and validation complete.** Added all-fold group-disjoint out-of-fold evaluation and an end-to-end CLI demonstration that runs generation, MNE conversion, spectral extraction, coherence, visualization, and model evaluation. The script writes report/figure outputs only to ignored `outputs/`; no signal data are serialized. The final full test suite, examples, notebook, compilation, Ruff, provenance, and repository hygiene checks passed.

## After the first week

1. Add behavioral/task-event streams with timestamp alignment and provenance.
2. Define collaboration-state labels through a defensible task protocol and annotation process before supervised learning.
3. Establish subject/session-aware baselines and leakage-resistant evaluation before multimodal modeling.
4. Explore PyTorch only when a question and dataset justify it; retain simple baselines.
5. Add a PyBullet task simulation and transparent assistance policy only after defining measurable task outcomes and safety constraints.
6. Treat interpersonal coherence/synchrony as statistical association requiring controls, uncertainty estimates, and cautious interpretation.

## Scope boundary

This plan does not authorize collection or processing of real human recordings. Before introducing human data, establish consent, privacy, secure handling, de-identification, data governance, and any applicable institutional review requirements. Synthetic performance is not evidence of human, clinical, or deployment performance.

## NeuroSync 2.0 — Seven-session Phase 2 delivery plan

The completed seven-day Phase 1 work above is retained. Phase 2 continues on the existing `main` history as seven new development sessions. Each session must end with tested, reviewed, documented work and one meaningful commit pushed to `origin/main`; if an external dependency prevents a milestone from being completed or tested, record the blocker without marking that milestone complete.

### Session 1 — Real PyBullet collaborative task simulation

- Establish a documented, reproducible PyBullet runtime on the target Windows/Python 3.11.9 environment or a documented WSL2/Linux route.
- Implement a headless shared-workspace task with two simulated agents, trajectories, distances, velocities, idle/hesitation telemetry, and scripted task-state events.
- Preserve engine identity and simulation provenance in every result. Mocked tests may exercise the interface, but cannot count as evidence that a real PyBullet simulation ran.

### Session 2 — Event-driven synthetic EEG and alignment

- Map task event timestamps to MNE annotations, events, and epochs with explicit clock offsets and boundary checks.
- Generate explicitly synthetic EEG whose injected shared component is controlled by scripted task phases. Preserve both simulation and signal-generation provenance.

### Session 3 — Unified feature extraction

- Extract EEG spectral/coherence and behavioral kinematic features from the same simulation runs.
- Define feature rows, units, event windows, missing-data behavior, and session/seed identifiers; test deterministic alignment and feature contracts.

### Session 4 — Comparative multimodal evaluation

- Compare EEG-only, behavior-only, and multimodal baselines using strict `StratifiedGroupKFold` grouping by simulation seed/run.
- Report balanced accuracy, macro F1, confusion matrices, class balance, uncertainty, and protocol limitations. Keep all preprocessing/model fitting within training folds.

### Session 5 — Adaptive assistance policy

- Implement an inspectable rule-based robot-assistance policy with logged inputs, decisions, and simulated behavior changes.
- Evaluate policy behavior against explicit scripted task conditions and baseline policies; make no real-robot safety claim.

### Session 6 — Interactive Streamlit research dashboard

- Provide an interactive browser dashboard for experiment status/provenance, simulation runs and trajectories, EEG signals/spectral/coherence views, comparative model results, and assistance decisions.
- Load reports from output directories, support run/channel/participant selection, and provide tests for data loading, missing reports, and dashboard calculations.

### Session 7 — End-to-end integration, CI, and demonstration

- Provide one master CLI that runs simulation through report generation, and a documented Windows launch path for the dashboard.
- Add GitHub Actions for Python 3.11, pytest, Ruff lint/format, and a real-engine pipeline smoke test; publish reports/figures as workflow artifacts.
- Verify reproducibility, provenance, output exclusions, and setup from a fresh documented environment.

### Current Phase 2 status (2026-11-18)

- Session 1's Linux PyBullet path is complete. Session 5 review caught a prior static agent B; both proxy spheres now have dynamic mass, and CI run `44355073362` passed the stronger real-engine checks requiring both bodies to move. Native Windows still lacks MSVC; WSL2 is not installed.
- Session 2 is complete. GitHub Actions run `44341146944` passed the actual PyBullet → event-driven synthetic EEG → MNE Raw annotations/epochs integration test. This confirms software timestamp mapping only, not a human physiological effect.
- Added `.github/workflows/ci.yml` to install PyBullet and run the simulation smoke example on Ubuntu/Python 3.11. Run `44161231064` succeeded after the dynamic-body update, including tests, quality checks, example execution, and an assertion that PyBullet-produced positions move. Native Windows remains unverified. The simulation is not a robot arm or human biomechanics model.
- Session 3 is complete. Unified event-level EEG spectral/coherence and behavioral kinematic features are implemented and tested. GitHub Actions run `44343178907` passed the real PyBullet feature example and test suite on Linux.
- Session 4 is complete. Group-aware EEG-only, behavior-only, and multimodal logistic models, out-of-fold reports, and confusion-matrix/metric figures pass local checks. GitHub Actions run `44344745446` passed the real-engine 12-run comparison example. The target is scripted task ground truth, not a human state.
- Session 5 is complete. The inspectable rule-based policy alters the dynamic proxy's target timing and control effort; paired evaluation and per-step inputs/actions are retained with simulation provenance. CI run `44355073362` passed the full checks, actual PyBullet policy example, and two-agent motion assertions.
- Session 6 dashboard implementation and local checks are complete: Streamlit views, validated manifest-based loading, reproducible example-data builder, data/calculation tests, and a CI smoke path. Commit/push and remote workflow verification are pending. Session 7 remains the full-pipeline CLI, report/figure artifacts, final documentation, and end-to-end verification. Every model target remains an authored simulation label.
