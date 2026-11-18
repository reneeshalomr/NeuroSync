"""Contracts for event-aligned synthetic EEG/simulation features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurosync.data.event_driven import generate_event_modulated_synthetic_eeg
from neurosync.features import extract_unified_features
from neurosync.simulation.task import CollaborativeTaskResult, TaskEvent


def _simulation(source: str = "simulated") -> CollaborativeTaskResult:
    sfreq = 20.0
    time = np.arange(121, dtype=float) / sfreq
    position_a = np.column_stack(
        (-0.6 + 0.1 * time, np.zeros_like(time), np.ones_like(time))
    )
    position_b = np.column_stack(
        (0.6 - 0.1 * time, np.zeros_like(time), np.ones_like(time))
    )
    velocity_a = np.tile([0.1, 0.0, 0.0], (time.size, 1))
    velocity_b = np.tile([-0.1, 0.0, 0.0], (time.size, 1))
    speed_a = np.linalg.norm(velocity_a, axis=1)
    speed_b = np.linalg.norm(velocity_b, axis=1)
    events = (
        TaskEvent(0, "approach", "synchronized", 0.0, 2.0),
        TaskEvent(1, "handover", "desynchronized", 2.0, 4.0),
        TaskEvent(2, "retreat", "synchronized", 4.0, 6.0),
    )
    return CollaborativeTaskResult(
        time_seconds=time,
        agent_a_position_m=position_a,
        agent_b_position_m=position_b,
        agent_a_velocity_m_s=velocity_a,
        agent_b_velocity_m_s=velocity_b,
        agent_a_speed_m_s=speed_a,
        agent_b_speed_m_s=speed_b,
        inter_agent_distance_m=np.linalg.norm(position_a - position_b, axis=1),
        events=events,
        velocity_correlation=0.0,
        idle_duration_seconds={"agent_a": 0.0, "agent_b": 0.0},
        metadata={
            "data_source": source,
            "duration_seconds": 6.0,
            "seed": 17,
            "engine": "fixture",
        },
    )


def _feature_result():
    sim = _simulation()
    eeg = generate_event_modulated_synthetic_eeg(
        sim, sampling_frequency_hz=128.0, n_channels=4, seed=18
    )
    return extract_unified_features(eeg, sim, session_id="fixture-session")


def test_unified_rows_have_aligned_eeg_behavior_features_and_provenance() -> None:
    result = _feature_result()
    assert len(result.rows) == 3
    assert list(result.rows["phase_name"]) == ["approach", "handover", "retreat"]
    assert list(result.rows["task_state"]) == [
        "synchronized",
        "desynchronized",
        "synchronized",
    ]
    assert set(result.rows["data_source_eeg"]) == {"synthetic"}
    assert set(result.rows["data_source_behavior"]) == {"simulated"}
    for name in (
        "eeg_a_mean_alpha_relative_power",
        "eeg_b_mean_gamma_power_v2",
        "eeg_mean_alpha_coherence",
        "mean_inter_agent_distance_m",
        "agent_a_path_length_m",
        "agent_b_max_speed_m_s",
    ):
        assert name in result.rows
    numeric = result.rows.select_dtypes(include=[np.number]).to_numpy()
    assert np.isfinite(numeric).all()
    assert not result.rows.isna().any().any()
    assert result.metadata["label_source"] == "scripted_simulation_ground_truth"
    assert result.metadata["interval_convention"] == "half-open [start, end)"


def test_unified_feature_extraction_is_reproducible() -> None:
    first = _feature_result()
    second = _feature_result()
    pd.testing.assert_frame_equal(first.rows, second.rows)
    assert first.metadata == second.metadata


def test_unified_features_reject_non_simulated_behavior() -> None:
    sim = _simulation(source="human")
    eeg = generate_event_modulated_synthetic_eeg(
        _simulation(), sampling_frequency_hz=128, n_channels=2, seed=1
    )
    with pytest.raises(ValueError, match="simulated"):
        extract_unified_features(eeg, sim)


def test_unified_features_reject_duration_mismatch() -> None:
    sim = _simulation()
    eeg = generate_event_modulated_synthetic_eeg(
        sim, sampling_frequency_hz=128, n_channels=2, seed=1
    )
    sim.metadata["duration_seconds"] = 5.0
    with pytest.raises(ValueError, match="durations"):
        extract_unified_features(eeg, sim)
