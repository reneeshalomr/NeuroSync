"""Run grouped EEG/behavior model comparison on real PyBullet-generated features."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from neurosync.data.event_driven import generate_event_modulated_synthetic_eeg
from neurosync.features import UnifiedFeatureTable, extract_unified_features
from neurosync.modeling import (
    MultimodalCVConfig,
    evaluate_multimodal_comparison,
)
from neurosync.simulation import CollaborativeTaskConfig, run_collaborative_task
from neurosync.visualization.model_comparison import plot_model_comparison


def main() -> None:
    run_count = 12
    tables = []
    for run_index in range(run_count):
        simulation_seed = 3100 + run_index
        simulation = run_collaborative_task(
            CollaborativeTaskConfig(seed=simulation_seed)
        )
        eeg = generate_event_modulated_synthetic_eeg(
            simulation,
            sampling_frequency_hz=128.0,
            n_channels=4,
            seed=simulation_seed + 10000,
        )
        tables.append(
            extract_unified_features(
                eeg,
                simulation,
                session_id=f"simulated-run-{simulation_seed}",
            ).rows
        )
    feature_table = UnifiedFeatureTable(
        rows=pd.concat(tables, ignore_index=True),
        metadata={
            "data_source_eeg": "synthetic",
            "data_source_behavior": "simulated",
            "label_source": "scripted_simulation_ground_truth",
            "run_count": run_count,
        },
    )
    result = evaluate_multimodal_comparison(
        feature_table,
        MultimodalCVConfig(n_splits=4, random_state=52),
    )
    output_dir = Path("outputs/multimodal_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "metrics": result.metrics,
        "fold_metrics": result.fold_metrics,
        "class_balance": result.class_balance,
        "evaluation_metadata": result.metadata,
        "data_source": "synthetic EEG + PyBullet-simulated behavior",
    }
    (output_dir / "synthetic_simulated_model_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    figure = plot_model_comparison(result)
    figure.savefig(output_dir / "synthetic_simulated_model_comparison.png", dpi=180)
    print("Synthetic EEG and simulated task-state model comparison only.")
    print(json.dumps(result.metrics, indent=2, sort_keys=True))
    print(f"Report and figure: {output_dir}")


if __name__ == "__main__":
    main()
