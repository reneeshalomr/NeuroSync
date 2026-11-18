"""Configuration and actual-engine integration tests for full pipeline output."""

from __future__ import annotations

import json

import pytest

from neurosync.dashboard.data import load_dashboard_data
from neurosync.pipeline import FullRunConfig, run_full_pipeline
from neurosync.preprocessing import PreprocessingConfig
from neurosync.simulation import task as simulation_task


def test_full_run_config_validates_group_splits_and_nyquist() -> None:
    with pytest.raises(ValueError, match="run_count"):
        FullRunConfig(run_count=3, cv_splits=4)
    with pytest.raises(ValueError, match="exceed 90.7"):
        FullRunConfig(eeg_sampling_frequency_hz=90.0)
    with pytest.raises(ValueError, match="below EEG Nyquist"):
        FullRunConfig(
            eeg_sampling_frequency_hz=100.0,
            preprocessing=PreprocessingConfig(lowpass_hz=50.0),
        )


@pytest.mark.skipif(
    simulation_task.bullet is None,
    reason="end-to-end pipeline requires real PyBullet",
)
def test_full_pipeline_creates_dashboard_ready_real_engine_artifacts(tmp_path) -> None:
    result = run_full_pipeline(
        FullRunConfig(
            output_dir=tmp_path / "outputs" / "full_run",
            run_count=4,
            seed_start=130,
            eeg_channel_count=2,
            cv_splits=4,
        )
    )
    assert result.run_count == 4
    assert result.report_path.is_file()
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["provenance"]["eeg"].startswith("synthetic")
    assert report["provenance"]["behavior"].startswith("real PyBullet")
    assert all(
        count == 3
        for participant_counts in report[
            "mne_epoch_counts_per_run_and_participant"
        ].values()
        for count in participant_counts.values()
    )

    dashboard = load_dashboard_data(result.output_dir)
    assert len(dashboard.runs) == 4
    assert dashboard.runs[0].simulation_metadata["engine"] == "PyBullet DIRECT"
    assert dashboard.runs[0].eeg_metadata["data_source"] == "synthetic"
    assert dashboard.model_report is not None
    assert dashboard.assistance_report is not None
    assert not dashboard.runs[0].assistance_decisions.empty
    assert (result.output_dir / "figures" / "synthetic_eeg_example.png").is_file()
    assert (
        result.output_dir / "figures" / "synthetic_simulated_model_comparison.png"
    ).is_file()
    assert (
        result.output_dir / "figures" / "simulated_assistance_comparison.png"
    ).is_file()
