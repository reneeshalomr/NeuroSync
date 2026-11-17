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

## Day 4 — Synchrony analysis

- Add descriptive inter-signal statistics with windowing, preprocessing, and surrogate/control assumptions stated.
- Test calculations on synthetic signals without treating constructed shared components as evidence about humans.

**Exit criteria:** outputs are described as statistical association; no coherence or phase metric is called proof of interpersonal synchrony.

## Day 5 — Visualization and exploratory research notebook

- Build reusable plots and a notebook calling package functions for synthetic signals, features, and synchrony summaries.
- Label simulated data, units, parameters, and limitations in every figure.

**Exit criteria:** the notebook runs reproducibly and separates exploration from confirmatory evaluation.

## Day 6 — Baseline ML for synthetic collaboration-state classification

- Define a clearly synthetic target and simple baseline models; use participant/session-aware splits where the data contract supports them.
- Report class balance, leakage controls, uncertainty, and evaluation limits.

**Exit criteria:** tests and metrics demonstrate software workflow only; no real-world validity claim is made.

## Day 7 — Integration, evaluation, documentation, and demonstration

- Run the end-to-end synthetic workflow, full test suite, compilation, and code-quality checks.
- Review provenance, generated-file exclusions, scientific language, and installation instructions.
- Summarize completed work, measured checks, remaining risks, and future research tasks.

**Exit criteria:** the staged synthetic platform is reproducible and reviewable; no real participant data or robot deployment claim is implied.

## After the first week

1. Add behavioral/task-event streams with timestamp alignment and provenance.
2. Define collaboration-state labels through a defensible task protocol and annotation process before supervised learning.
3. Establish subject/session-aware baselines and leakage-resistant evaluation before multimodal modeling.
4. Explore PyTorch only when a question and dataset justify it; retain simple baselines.
5. Add a PyBullet task simulation and transparent assistance policy only after defining measurable task outcomes and safety constraints.
6. Treat interpersonal coherence/synchrony as statistical association requiring controls, uncertainty estimates, and cautious interpretation.

## Scope boundary

This plan does not authorize collection or processing of real human recordings. Before introducing human data, establish consent, privacy, secure handling, de-identification, data governance, and any applicable institutional review requirements. Synthetic performance is not evidence of human, clinical, or deployment performance.
