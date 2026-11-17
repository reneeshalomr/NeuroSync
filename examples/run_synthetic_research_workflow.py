"""Run the end-to-end synthetic EEG analysis and grouped baseline workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from neurosync.collaboration.synchrony import compute_synthetic_coherence
from neurosync.data.synthetic_eeg import (
    ExperimentalCondition,
    generate_synthetic_eeg,
)
from neurosync.features.spectral import extract_spectral_features
from neurosync.modeling.synthetic_baseline import (
    build_synthetic_condition_dataset,
    evaluate_synthetic_baselines_cv,
)
from neurosync.preprocessing.raw import synthetic_participant_to_raw
from neurosync.visualization import (
    plot_coherence,
    plot_spectral_features,
    plot_synthetic_signals,
)


def run_workflow(output_dir: Path, *, n_sessions: int = 16) -> dict[str, Any]:
    """Generate a deterministic synthetic analysis and save ignored outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = generate_synthetic_eeg(
        sampling_frequency_hz=256,
        duration_seconds=20,
        n_channels=4,
        seed=2027,
        condition=ExperimentalCondition.HIGH_SHARED_COMPONENT,
    )
    raw_a = synthetic_participant_to_raw(data, "A")
    raw_b = synthetic_participant_to_raw(data, "B")
    spectral = extract_spectral_features(raw_a)
    coherence = compute_synthetic_coherence(raw_a, raw_b)

    figures = {
        "signals_synthetic.png": plot_synthetic_signals(
            data, participant="A", channels=(0, 1), duration_seconds=5
        ),
        "spectral_synthetic.png": plot_spectral_features(spectral),
        "coherence_synthetic.png": plot_coherence(coherence),
    }
    for filename, figure in figures.items():
        figure.savefig(output_dir / filename, dpi=140, bbox_inches="tight")
        plt.close(figure)

    dataset = build_synthetic_condition_dataset(
        n_sessions=n_sessions,
        seed=6000,
        sampling_frequency_hz=256,
        duration_seconds=8,
        n_channels=4,
    )
    evaluation = evaluate_synthetic_baselines_cv(
        dataset,
        random_state=29,
        n_splits=4,
    )
    report = {
        "scientific_status": "synthetic software workflow; not human EEG or validated model results",
        "signal_demo": {
            "data_source": data.data_source,
            "condition": data.condition.value,
            "seed": data.random_seed,
            "sampling_frequency_hz": data.sampling_frequency_hz,
            "duration_seconds": float(
                data.time_seconds.size / data.sampling_frequency_hz
            ),
            "channel_names": list(data.channel_names),
            "signal_unit": "V",
            "spectral_feature_metadata": spectral.metadata,
            "coherence_metadata": coherence.metadata,
            "mean_coherence_by_band": coherence.band_mean_coherence,
        },
        "synthetic_condition_evaluation": {
            "class_balance": evaluation.class_balance,
            "fold_test_groups": [
                list(groups) for groups in evaluation.fold_test_groups
            ],
            "fold_metrics": {
                name: list(fold_values)
                for name, fold_values in evaluation.fold_metrics.items()
            },
            "aggregate_metrics": evaluation.aggregate_metrics,
            "metadata": evaluation.metadata,
        },
        "generated_figures": list(figures),
        "interpretation_limit": (
            "The classifier target is the synthetic generator setting, not an annotated human collaboration state. "
            "Coherence and synthetic model scores do not establish interpersonal synchrony or human-state validity."
        ),
    }
    report_path = output_dir / "synthetic_workflow_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Synthetic-only workflow complete; report: {report_path}")
    for name, metrics in evaluation.aggregate_metrics.items():
        print(
            f"{name}: out-of-fold accuracy={metrics['accuracy']:.3f}, "
            f"balanced accuracy={metrics['balanced_accuracy']:.3f}, "
            f"macro F1={metrics['macro_f1']:.3f}, "
            f"paired-group exact accuracy={metrics['paired_group_exact_accuracy']:.3f}"
        )
    print(
        "Scores describe this constructed generator target only; no human validity is implied."
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/synthetic_research_demo"),
        help="directory for generated PNG figures and JSON report (default: ignored outputs/)",
    )
    args = parser.parse_args()
    run_workflow(args.output_dir)


if __name__ == "__main__":
    main()
