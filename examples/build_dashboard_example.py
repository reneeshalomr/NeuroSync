"""Build a reproducible dashboard dataset from real PyBullet and synthetic EEG."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neurosync.data.event_driven import generate_event_modulated_synthetic_eeg
from neurosync.features import UnifiedFeatureTable, extract_unified_features
from neurosync.modeling import MultimodalCVConfig, evaluate_multimodal_comparison
from neurosync.simulation import (
    CollaborativeTaskConfig,
    evaluate_rule_based_assistance,
    run_collaborative_task,
    save_simulation_result,
)


def _save_eeg(eeg, path: Path) -> None:
    metadata = {
        "data_source": "synthetic",
        "signal_unit": "V",
        "condition": eeg.condition.value,
        "random_seed": eeg.random_seed,
        "sampling_frequency_hz": eeg.sampling_frequency_hz,
        "generation_parameters": eeg.generation_parameters,
        "simulation_metadata": eeg.simulation_metadata,
        "limitations": "Constructed EEG-like data, not real human recordings or a validated physiological model.",
    }
    np.savez_compressed(
        path,
        time_seconds=eeg.time_seconds,
        participant_a=eeg.participant_a,
        participant_b=eeg.participant_b,
        channel_names=np.asarray(eeg.channel_names),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def main() -> None:
    output = Path("outputs/dashboard_example")
    output.mkdir(parents=True, exist_ok=True)
    tables: list[pd.DataFrame] = []
    run_specs: list[dict[str, str]] = []
    assistance_evaluation = evaluate_rule_based_assistance(
        CollaborativeTaskConfig(seed=7200)
    )

    for index in range(12):
        seed = 7200 + index
        if index == 0:
            simulation = assistance_evaluation.adaptive
            assistance_file = "assisted_simulated_seed7200.npz"
            save_simulation_result(simulation, output / assistance_file)
            decisions_file = "assisted_simulated_seed7200_decisions.csv"
            pd.DataFrame(simulation.assistance_decisions).to_csv(
                output / decisions_file, index=False
            )
        else:
            simulation = run_collaborative_task(CollaborativeTaskConfig(seed=seed))
            assistance_file = ""
            decisions_file = ""
            save_simulation_result(simulation, output / f"run_{seed}_simulated.npz")

        eeg = generate_event_modulated_synthetic_eeg(
            simulation,
            sampling_frequency_hz=128.0,
            n_channels=4,
            seed=seed + 20000,
        )
        eeg_file = f"run_{seed}_synthetic_eeg.npz"
        _save_eeg(eeg, output / eeg_file)
        features = extract_unified_features(
            eeg, simulation, session_id=f"dashboard-example-{seed}"
        )
        feature_file = f"run_{seed}_synthetic_simulated_features.csv"
        features.rows.to_csv(output / feature_file, index=False)
        tables.append(features.rows)
        run_specs.append(
            {
                "run_id": f"simulated-seed-{seed}",
                "simulation_file": assistance_file or f"run_{seed}_simulated.npz",
                "eeg_file": eeg_file,
                "features_file": feature_file,
                **({"assistance_file": decisions_file} if decisions_file else {}),
            }
        )

    combined = UnifiedFeatureTable(
        rows=pd.concat(tables, ignore_index=True),
        metadata={
            "data_source_eeg": "synthetic",
            "data_source_behavior": "simulated",
            "label_source": "scripted_simulation_ground_truth",
        },
    )
    models = evaluate_multimodal_comparison(
        combined, MultimodalCVConfig(n_splits=4, random_state=72)
    )
    model_report = {
        "metrics": models.metrics,
        "class_balance": models.class_balance,
        "evaluation_metadata": models.metadata,
        "provenance": "synthetic EEG + simulated PyBullet features and scripted labels",
    }
    (output / "synthetic_simulated_model_report.json").write_text(
        json.dumps(model_report, indent=2, sort_keys=True), encoding="utf-8"
    )
    assistance_report = {
        "metrics": assistance_evaluation.metrics,
        "evaluation_metadata": assistance_evaluation.metadata,
    }
    (output / "simulated_assistance_report.json").write_text(
        json.dumps(assistance_report, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "data_source": "synthetic_and_simulated",
        "dataset_name": "reproducible_dashboard_example",
        "random_seed_range": [7200, 7211],
        "run_count": len(run_specs),
        "model_report_file": "synthetic_simulated_model_report.json",
        "assistance_report_file": "simulated_assistance_report.json",
        "runs": run_specs,
        "limitations": [
            "EEG is software-generated and not a human recording.",
            "Task-state labels are authored simulation ground truth.",
            "All model scores describe this constructed dataset only.",
        ],
    }
    (output / "dashboard_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("Created a reproducible synthetic/simulation dashboard example.")
    print(f"Runs: {len(run_specs)}; simulated seeds: 7200–7211")
    print(f"Dashboard data: {output.resolve()}")


if __name__ == "__main__":
    main()
