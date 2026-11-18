"""Dashboard artifact contract and calculation tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from neurosync.dashboard.data import (
    DashboardDataError,
    confusion_matrix_frame,
    load_dashboard_data,
    summarize_band_power,
    summarize_model_metrics,
)


def _write_bundle(root, *, model_report=True):
    times = np.arange(3, dtype=float) * 0.1
    events = [
        {
            "event_id": 1,
            "phase_name": "test",
            "task_state": "synchronized",
            "start_time_seconds": 0.0,
            "end_time_seconds": 0.2,
        }
    ]
    sim_metadata = {"data_source": "simulated", "engine": "PyBullet DIRECT", "seed": 19}
    np.savez_compressed(
        root / "run_simulated.npz",
        time_seconds=times,
        agent_a_position_m=np.zeros((3, 3)),
        agent_b_position_m=np.ones((3, 3)),
        agent_a_velocity_m_s=np.zeros((3, 3)),
        agent_b_velocity_m_s=np.ones((3, 3)),
        agent_a_speed_m_s=np.zeros(3),
        agent_b_speed_m_s=np.ones(3),
        inter_agent_distance_m=np.ones(3),
        events_json=np.asarray(json.dumps(events)),
        metadata_json=np.asarray(json.dumps(sim_metadata)),
        assistance_decisions_json=np.asarray("[]"),
    )
    eeg_metadata = {
        "data_source": "synthetic",
        "signal_unit": "V",
        "sampling_frequency_hz": 10,
    }
    np.savez_compressed(
        root / "eeg_synthetic.npz",
        time_seconds=times,
        participant_a=np.ones((1, 3)) * 1e-6,
        participant_b=np.ones((1, 3)) * 2e-6,
        channel_names=np.asarray(["EEG001"]),
        metadata_json=np.asarray(json.dumps(eeg_metadata)),
    )
    features = {
        "task_state": ["synchronized"],
        "phase_name": ["test"],
        "event_start_seconds": [0.0],
        "event_end_seconds": [0.2],
    }
    for participant in ("a", "b"):
        for index, band in enumerate(
            ("delta", "theta", "alpha", "beta", "gamma"), start=1
        ):
            features[f"eeg_{participant}_mean_{band}_power_v2"] = [float(index * 1e-12)]
    pd.DataFrame(features).to_csv(root / "features.csv", index=False)
    manifest = {
        "runs": [
            {
                "run_id": "test-seed-19",
                "simulation_file": "run_simulated.npz",
                "eeg_file": "eeg_synthetic.npz",
                "features_file": "features.csv",
            }
        ]
    }
    if model_report:
        manifest["model_report_file"] = "model_report.json"
        report = {
            "metrics": {
                "eeg_only": {
                    "balanced_accuracy": 0.5,
                    "macro_f1": 0.4,
                    "class_order": ["sync", "desync"],
                    "confusion_matrix": [[2, 1], [1, 2]],
                },
                "behavior_only": {"balanced_accuracy": 0.6, "macro_f1": 0.55},
            },
            "evaluation_metadata": {
                "target_source": "scripted_simulation_ground_truth"
            },
        }
        (root / "model_report.json").write_text(json.dumps(report), encoding="utf-8")
    (root / "dashboard_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_loads_pickle_free_dashboard_bundle_and_model_report(tmp_path) -> None:
    _write_bundle(tmp_path)
    data = load_dashboard_data(tmp_path)
    assert len(data.runs) == 1
    assert data.runs[0].run_id == "test-seed-19"
    assert data.runs[0].eeg_metadata["data_source"] == "synthetic"
    assert data.runs[0].simulation_metadata["engine"] == "PyBullet DIRECT"
    assert data.model_report is not None
    assert data.runs[0].eeg["participant_a"].shape == (1, 3)


def test_missing_manifest_explains_how_to_generate_example(tmp_path) -> None:
    with pytest.raises(DashboardDataError, match="build_dashboard_example.py"):
        load_dashboard_data(tmp_path)


def test_missing_optional_reports_are_exposed_without_breaking_run_views(
    tmp_path,
) -> None:
    _write_bundle(tmp_path, model_report=False)
    data = load_dashboard_data(tmp_path)
    assert data.model_report is None
    assert data.assistance_report is None


def test_missing_required_artifact_and_path_traversal_are_rejected(tmp_path) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "eeg_synthetic.npz").unlink()
    with pytest.raises(DashboardDataError, match="Missing synthetic EEG"):
        load_dashboard_data(tmp_path)

    _write_bundle(tmp_path)
    manifest_path = tmp_path / "dashboard_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runs"][0]["eeg_file"] = "../private.npz"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DashboardDataError, match="stay under output root"):
        load_dashboard_data(tmp_path)


def test_model_metric_and_band_power_calculations() -> None:
    report = {"metrics": {"eeg_only": {"balanced_accuracy": 0.6, "macro_f1": 0.55}}}
    summary = summarize_model_metrics(report)
    assert summary.iloc[0]["balanced_accuracy"] == pytest.approx(0.6)
    assert summary.iloc[0]["macro_f1"] == pytest.approx(0.55)
    feature_values = {
        f"eeg_a_mean_{band}_power_v2": [index, index + 2]
        for index, band in enumerate(
            ("delta", "theta", "alpha", "beta", "gamma"), start=1
        )
    }
    bands = summarize_band_power(pd.DataFrame(feature_values), "a")
    assert bands["absolute_power_v2"].to_list() == pytest.approx([2, 3, 4, 5, 6])
    matrix = confusion_matrix_frame(
        {"class_order": ["a", "b"], "confusion_matrix": [[2, 1], [0, 3]]}
    )
    assert matrix.loc["a", "b"] == 1


def test_dashboard_calculations_reject_invalid_metrics_and_matrices() -> None:
    with pytest.raises(DashboardDataError, match="invalid balanced_accuracy"):
        summarize_model_metrics(
            {"metrics": {"bad": {"balanced_accuracy": float("nan"), "macro_f1": 0.5}}}
        )
    with pytest.raises(DashboardDataError, match="dimensions or values"):
        confusion_matrix_frame({"class_order": ["a", "b"], "confusion_matrix": [[1]]})
