"""CLI for simulation-to-dashboard NeuroSync research pipeline execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neurosync.pipeline import FullRunConfig, run_full_pipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run real PyBullet simulation, event-aligned synthetic EEG, MNE "
            "preprocessing/epochs, unified feature extraction, group-aware models, "
            "adaptive assistance, and dashboard artifact generation."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/full_run"))
    parser.add_argument("--run-count", type=int, default=12)
    parser.add_argument("--seed-start", type=int, default=9100)
    parser.add_argument("--eeg-sampling-frequency-hz", type=float, default=128.0)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--cv-splits", type=int, default=4)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_full_pipeline(
        FullRunConfig(
            output_dir=args.output_dir,
            run_count=args.run_count,
            seed_start=args.seed_start,
            eeg_sampling_frequency_hz=args.eeg_sampling_frequency_hz,
            eeg_channel_count=args.channels,
            cv_splits=args.cv_splits,
        )
    )
    print(
        json.dumps(
            {
                "status": result.report["status"],
                "run_count": result.run_count,
                "output_dir": str(result.output_dir),
                "summary_report": str(result.report_path),
                "dashboard_manifest": str(result.manifest_path),
                "model_metrics": result.report["model_metrics"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
