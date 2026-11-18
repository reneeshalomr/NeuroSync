"""Run real PyBullet simulation and extract aligned synthetic EEG/behavior features."""

from __future__ import annotations

import json
from pathlib import Path

from neurosync.data.event_driven import generate_event_modulated_synthetic_eeg
from neurosync.features import extract_unified_features
from neurosync.simulation import CollaborativeTaskConfig, run_collaborative_task


def main() -> None:
    simulation = run_collaborative_task(CollaborativeTaskConfig(seed=2026))
    eeg = generate_event_modulated_synthetic_eeg(
        simulation, sampling_frequency_hz=256.0, n_channels=8, seed=72
    )
    feature_table = extract_unified_features(
        eeg, simulation, session_id="simulated-seed-2026"
    )
    output = Path("outputs/simulation_features")
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "synthetic_simulated_unified_features.csv"
    metadata_path = output / "synthetic_simulated_feature_metadata.json"
    feature_table.rows.to_csv(csv_path, index=False)
    metadata_path.write_text(
        json.dumps(feature_table.metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("Synthetic EEG and simulated behavior feature extraction only.")
    print(
        f"Rows: {len(feature_table.rows)}; columns: {len(feature_table.rows.columns)}"
    )
    print(f"Features: {csv_path}")
    print(f"Provenance: {metadata_path}")


if __name__ == "__main__":
    main()
