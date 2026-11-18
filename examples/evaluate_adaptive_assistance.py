"""Compare a logged rule-based robot-proxy policy with the unassisted task."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from neurosync.simulation import (
    CollaborativeTaskConfig,
    evaluate_rule_based_assistance,
    save_simulation_result,
)
from neurosync.visualization.assistance import plot_assistance_evaluation


def main() -> None:
    evaluation = evaluate_rule_based_assistance(CollaborativeTaskConfig(seed=2026))
    output_dir = Path("outputs/adaptive_assistance")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "metrics": evaluation.metrics,
        "evaluation_metadata": evaluation.metadata,
        "adaptive_simulation_metadata": evaluation.adaptive.metadata,
    }
    (output_dir / "simulated_assistance_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(evaluation.adaptive.assistance_decisions).to_csv(
        output_dir / "simulated_assistance_decisions.csv", index=False
    )
    save_simulation_result(
        evaluation.unassisted,
        output_dir / "unassisted_simulated_seed2026.npz",
    )
    save_simulation_result(
        evaluation.adaptive,
        output_dir / "adaptive_simulated_seed2026.npz",
    )
    figure = plot_assistance_evaluation(evaluation)
    figure.savefig(output_dir / "simulated_assistance_comparison.png", dpi=180)
    print("Simulation-only rule-based assistance evaluation.")
    print(json.dumps(evaluation.metrics, indent=2, sort_keys=True))
    print(f"Policy decisions logged: {len(evaluation.adaptive.assistance_decisions)}")
    print(f"Report, telemetry, and figure: {output_dir}")


if __name__ == "__main__":
    main()
