"""Orchestrate real PyBullet runs through synthetic EEG, features, and evaluation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from neurosync.alignment import annotate_raw_with_task_events, epoch_task_events
from neurosync.data.event_driven import generate_event_modulated_synthetic_eeg
from neurosync.features import UnifiedFeatureTable, extract_unified_features
from neurosync.modeling import MultimodalCVConfig, evaluate_multimodal_comparison
from neurosync.preprocessing import (
    PreprocessingConfig,
    filter_eeg,
    synthetic_participant_to_raw,
)
from neurosync.simulation import (
    CollaborativeTaskConfig,
    evaluate_rule_based_assistance,
    run_collaborative_task,
    save_simulation_result,
)
from neurosync.visualization.assistance import plot_assistance_evaluation
from neurosync.visualization.model_comparison import plot_model_comparison
from neurosync.visualization.signals import plot_synthetic_signals


@dataclass(frozen=True)
class FullRunConfig:
    """Explicit settings for an end-to-end, simulation-only experiment."""

    output_dir: Path = Path("outputs/full_run")
    run_count: int = 12
    seed_start: int = 9100
    eeg_sampling_frequency_hz: float = 128.0
    eeg_channel_count: int = 4
    cv_splits: int = 4
    cv_random_state: int = 91
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    def __post_init__(self) -> None:
        if isinstance(self.cv_splits, bool) or not isinstance(self.cv_splits, int):
            raise TypeError("cv_splits must be an integer")
        if isinstance(self.run_count, bool) or not isinstance(self.run_count, int):
            raise TypeError("run_count must be an integer")
        if self.run_count < self.cv_splits or self.cv_splits < 2:
            raise ValueError("run_count must be at least cv_splits, and cv_splits >= 2")
        if isinstance(self.seed_start, bool) or not isinstance(self.seed_start, int):
            raise TypeError("seed_start must be an integer")
        if isinstance(self.cv_random_state, bool) or not isinstance(
            self.cv_random_state, int
        ):
            raise TypeError("cv_random_state must be an integer")
        if (
            isinstance(self.eeg_sampling_frequency_hz, bool)
            or not math.isfinite(self.eeg_sampling_frequency_hz)
            or self.eeg_sampling_frequency_hz <= 90.7
        ):
            raise ValueError(
                "EEG sampling frequency must exceed 90.7 Hz for generator bands"
            )
        nyquist = self.eeg_sampling_frequency_hz / 2.0
        for cutoff in (self.preprocessing.highpass_hz, self.preprocessing.lowpass_hz):
            if cutoff is not None and cutoff >= nyquist:
                raise ValueError(
                    "preprocessing cutoffs must be below EEG Nyquist frequency"
                )
        if (
            isinstance(self.eeg_channel_count, bool)
            or not isinstance(self.eeg_channel_count, int)
            or self.eeg_channel_count < 2
        ):
            raise ValueError("eeg_channel_count must be an integer of at least 2")


@dataclass(frozen=True)
class FullRunResult:
    """Locations and scientific provenance of one completed full run."""

    output_dir: Path
    report_path: Path
    manifest_path: Path
    run_count: int
    report: dict[str, Any]


def _save_eeg(eeg, path: Path, *, preprocessing: dict[str, Any]) -> None:
    metadata = {
        "data_source": "synthetic",
        "signal_unit": "V",
        "condition": eeg.condition.value,
        "random_seed": eeg.random_seed,
        "sampling_frequency_hz": eeg.sampling_frequency_hz,
        "generation_parameters": eeg.generation_parameters,
        "preprocessing": preprocessing,
        "simulation_metadata": eeg.simulation_metadata,
        "limitations": "Constructed EEG-like signals; not human recordings or validated physiology.",
    }
    np.savez_compressed(
        path,
        time_seconds=eeg.time_seconds,
        participant_a=eeg.participant_a,
        participant_b=eeg.participant_b,
        channel_names=np.asarray(eeg.channel_names),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    temporary_path.replace(path)


def run_full_pipeline(config: FullRunConfig | None = None) -> FullRunResult:
    """Run simulation → MNE event alignment/preprocessing → features → models.

    The first run is a matched paired policy evaluation. Its adaptive trajectory
    is retained for the dashboard; the remaining runs are unassisted. Model
    evaluation is grouped by run seed. Every label is authored task ground truth.
    """
    selected = config or FullRunConfig()
    output = Path(selected.output_dir).expanduser().resolve()
    figures_dir = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tables: list[pd.DataFrame] = []
    run_specs: list[dict[str, str]] = []
    epoch_counts: dict[str, dict[str, int]] = {}
    assistance_evaluation = evaluate_rule_based_assistance(
        CollaborativeTaskConfig(seed=selected.seed_start)
    )

    for index in range(selected.run_count):
        seed = selected.seed_start + index
        if index == 0:
            simulation = assistance_evaluation.adaptive
        else:
            simulation = run_collaborative_task(CollaborativeTaskConfig(seed=seed))

        simulation_file = f"run_seed{seed}_simulated.npz"
        save_simulation_result(simulation, output / simulation_file)
        eeg = generate_event_modulated_synthetic_eeg(
            simulation,
            sampling_frequency_hz=selected.eeg_sampling_frequency_hz,
            n_channels=selected.eeg_channel_count,
            seed=seed + 10000,
        )

        processed: dict[str, mne.io.BaseRaw] = {}
        epoch_counts[f"seed-{seed}"] = {}
        for participant in ("A", "B"):
            raw = synthetic_participant_to_raw(eeg, participant)
            aligned = annotate_raw_with_task_events(raw, simulation)
            epochs = epoch_task_events(raw, simulation, tmin=0.0, tmax=1.0)
            filtered = filter_eeg(aligned, selected.preprocessing)
            processed[participant] = filtered
            epoch_counts[f"seed-{seed}"][participant] = len(epochs)

        provenance = json.loads(processed["A"].info["description"])
        preprocessed_eeg = replace(
            eeg,
            participant_a=processed["A"].get_data(picks="eeg"),
            participant_b=processed["B"].get_data(picks="eeg"),
            generation_parameters={
                **eeg.generation_parameters,
                "signal_unit": "V",
                "preprocessing": provenance["preprocessing"],
            },
        )
        features = extract_unified_features(
            preprocessed_eeg,
            simulation,
            session_id=f"full-run-{seed}",
        )
        feature_file = f"run_seed{seed}_synthetic_simulated_features.csv"
        features.rows.to_csv(output / feature_file, index=False)
        tables.append(features.rows)

        eeg_file = f"run_seed{seed}_synthetic_eeg.npz"
        _save_eeg(
            preprocessed_eeg,
            output / eeg_file,
            preprocessing=provenance["preprocessing"],
        )
        run_spec = {
            "run_id": f"simulated-seed-{seed}",
            "simulation_file": simulation_file,
            "eeg_file": eeg_file,
            "features_file": feature_file,
        }
        if index == 0:
            decisions_file = f"run_seed{seed}_simulated_assistance_decisions.csv"
            pd.DataFrame(simulation.assistance_decisions).to_csv(
                output / decisions_file, index=False
            )
            run_spec["assistance_file"] = decisions_file
        run_specs.append(run_spec)

    combined = UnifiedFeatureTable(
        rows=pd.concat(tables, ignore_index=True),
        metadata={
            "data_source_eeg": "synthetic",
            "data_source_behavior": "simulated",
            "label_source": "scripted_simulation_ground_truth",
        },
    )
    models = evaluate_multimodal_comparison(
        combined,
        MultimodalCVConfig(
            n_splits=selected.cv_splits,
            random_state=selected.cv_random_state,
        ),
    )
    model_report = {
        "metrics": models.metrics,
        "fold_metrics": models.fold_metrics,
        "class_balance": models.class_balance,
        "evaluation_metadata": models.metadata,
        "data_source": "synthetic EEG + simulated PyBullet behavior",
    }
    model_path = output / "synthetic_simulated_model_report.json"
    _write_json(model_path, model_report)

    assistance_report = {
        "metrics": assistance_evaluation.metrics,
        "evaluation_metadata": assistance_evaluation.metadata,
    }
    assistance_path = output / "simulated_assistance_report.json"
    _write_json(assistance_path, assistance_report)

    model_figure = plot_model_comparison(models)
    model_figure.savefig(
        figures_dir / "synthetic_simulated_model_comparison.png", dpi=180
    )
    signal_figure = plot_synthetic_signals(
        preprocessed_eeg,
        participant="A",
        duration_seconds=3.0,
    )
    signal_figure.savefig(figures_dir / "synthetic_eeg_example.png", dpi=180)
    assistance_figure = plot_assistance_evaluation(assistance_evaluation)
    assistance_figure.savefig(
        figures_dir / "simulated_assistance_comparison.png", dpi=180
    )
    import matplotlib.pyplot as plt

    for figure in (model_figure, signal_figure, assistance_figure):
        plt.close(figure)

    report = {
        "project": "NeuroSync 2.0 full automated research pipeline",
        "data_source": "synthetic EEG + PyBullet-simulated behavior",
        "status": "completed",
        "config": {
            **asdict(selected),
            "output_dir": str(output),
            "preprocessing": asdict(selected.preprocessing),
        },
        "simulation_seeds": list(
            range(selected.seed_start, selected.seed_start + selected.run_count)
        ),
        "mne_epoch_counts_per_run_and_participant": epoch_counts,
        "model_metrics": models.metrics,
        "model_class_balance": models.class_balance,
        "assistance_metrics": assistance_evaluation.metrics,
        "provenance": {
            "eeg": "synthetic; signal values in volts; no human recordings",
            "behavior": "real PyBullet DIRECT dynamics of scripted proxies",
            "labels": "scripted_simulation_ground_truth",
            "group_cv": "StratifiedGroupKFold grouped by session and simulation seed",
        },
        "limitations": [
            "Task actors are scripted dynamic proxies, not human or robot-arm models.",
            "EEG-like data are constructed signals, not human recordings or validated physiology.",
            "Task-state labels are authored simulation ground truth rather than inferred human collaboration states.",
            "Model scores describe this synthetic/simulated protocol only and do not establish human-state prediction validity.",
            "Coherence is descriptive statistical association; it does not prove interpersonal synchrony.",
            "The assistance demonstration does not establish human benefit, physical safety, or deployment readiness.",
        ],
        "artifacts": {
            "model_report": model_path.name,
            "assistance_report": assistance_path.name,
            "figures": [
                "figures/synthetic_simulated_model_comparison.png",
                "figures/synthetic_eeg_example.png",
                "figures/simulated_assistance_comparison.png",
            ],
        },
    }
    report_path = output / "synthetic_simulated_full_run_report.json"
    _write_json(report_path, report)
    manifest = {
        "data_source": "synthetic_and_simulated",
        "dataset_name": "neurosync_full_run",
        "run_count": len(run_specs),
        "model_report_file": model_path.name,
        "assistance_report_file": assistance_path.name,
        "full_run_report_file": report_path.name,
        "runs": run_specs,
        "limitations": report["limitations"],
    }
    manifest_path = output / "dashboard_manifest.json"
    _write_json(manifest_path, manifest)
    return FullRunResult(output, report_path, manifest_path, len(run_specs), report)
