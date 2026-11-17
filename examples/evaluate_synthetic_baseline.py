"""Run a grouped baseline on the synthetic generator-condition target."""

from __future__ import annotations

from neurosync.modeling.synthetic_baseline import (
    build_synthetic_condition_dataset,
    evaluate_synthetic_baselines,
)


def main() -> None:
    dataset = build_synthetic_condition_dataset(
        n_sessions=12,
        seed=2026,
        sampling_frequency_hz=256,
        duration_seconds=8,
        n_channels=4,
    )
    result = evaluate_synthetic_baselines(dataset, random_state=7, n_splits=4)
    print(
        "Synthetic generator-setting classification only; not human collaboration state"
    )
    print(f"Target: {result.metadata['target_definition']}")
    print(f"Class balance: {result.class_balance}")
    print(
        f"Training seed groups: {len(result.train_groups)}; held-out groups: {len(result.test_groups)}"
    )
    for model, metrics in result.metrics.items():
        print(
            f"{model}: accuracy={metrics['accuracy']:.3f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.3f}, "
            f"macro_f1={metrics['macro_f1']:.3f}, "
            f"paired-group all-rows-correct Wilson interval="
            f"{metrics['paired_group_exact_accuracy_95pct_wilson_interval']}"
        )
    print(
        "Metrics check the synthetic software workflow and do not indicate real-world validity or human-state predictability."
    )


if __name__ == "__main__":
    main()
