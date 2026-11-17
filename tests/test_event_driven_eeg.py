"""Tests for event-modulated synthetic signal construction and provenance."""

from __future__ import annotations

import json

import numpy as np
import pytest

from neurosync.data.event_driven import (
    DESYNCHRONIZED_COUPLING_SCALE,
    SYNCHRONIZED_COUPLING_SCALE,
    generate_event_modulated_synthetic_eeg,
)
from neurosync.data.synthetic_eeg import generate_synthetic_eeg
from neurosync.preprocessing.raw import synthetic_participant_to_raw
from neurosync.simulation.task import CollaborativeTaskResult, TaskEvent


def _simulation(*, source: str = "simulated", gap: bool = False):
    sfreq = 20.0
    times = np.arange(121, dtype=np.float64) / sfreq
    events = (
        TaskEvent(0, "approach", "synchronized", 0.0, 2.0),
        TaskEvent(1, "handover", "desynchronized", 2.0, 4.0),
        TaskEvent(2, "retreat", "synchronized", 4.0, 6.0),
    )
    if gap:
        events = (
            TaskEvent(0, "approach", "synchronized", 0.0, 1.5),
            TaskEvent(1, "retreat", "synchronized", 4.0, 6.0),
        )
    empty = np.zeros((times.size, 3), dtype=np.float64)
    speed = np.zeros(times.size, dtype=np.float64)
    return CollaborativeTaskResult(
        time_seconds=times,
        agent_a_position_m=empty,
        agent_b_position_m=empty.copy(),
        agent_a_velocity_m_s=empty.copy(),
        agent_b_velocity_m_s=empty.copy(),
        agent_a_speed_m_s=speed,
        agent_b_speed_m_s=speed.copy(),
        inter_agent_distance_m=speed.copy(),
        events=events,
        velocity_correlation=0.0,
        idle_duration_seconds={"agent_a": 6.0, "agent_b": 6.0},
        metadata={
            "data_source": source,
            "duration_seconds": 6.0,
            "seed": 12,
            "engine": "fixture",
        },
    )


def test_event_envelope_scales_shared_component_by_task_state() -> None:
    sim = _simulation()
    result = generate_event_modulated_synthetic_eeg(
        sim, sampling_frequency_hz=128, n_channels=2, seed=51
    )
    base = generate_synthetic_eeg(
        sampling_frequency_hz=128,
        duration_seconds=6,
        n_channels=2,
        seed=51,
    )
    synchronized = result.sample_task_state == "synchronized"
    desynchronized = result.sample_task_state == "desynchronized"
    assert synchronized.sum() > 0 and desynchronized.sum() > 0
    np.testing.assert_allclose(
        result.shared_component[synchronized],
        base.shared_component[synchronized]
        * SYNCHRONIZED_COUPLING_SCALE
        / base.generation_parameters["shared_component_scale"],
    )
    np.testing.assert_allclose(
        result.shared_component[desynchronized],
        base.shared_component[desynchronized]
        * DESYNCHRONIZED_COUPLING_SCALE
        / base.generation_parameters["shared_component_scale"],
    )
    assert result.data_source == "synthetic"
    assert result.simulation_metadata["data_source"] == "simulated"
    assert (
        "not model or establish human neural synchrony"
        in result.generation_parameters["interpretation_limit"]
    )


def test_event_driven_signals_and_metadata_are_reproducible() -> None:
    simulation = _simulation()
    first = generate_event_modulated_synthetic_eeg(
        simulation, sampling_frequency_hz=128, n_channels=2, seed=7
    )
    second = generate_event_modulated_synthetic_eeg(
        simulation, sampling_frequency_hz=128, n_channels=2, seed=7
    )
    np.testing.assert_array_equal(first.participant_a, second.participant_a)
    np.testing.assert_array_equal(first.participant_b, second.participant_b)
    np.testing.assert_array_equal(first.sample_task_state, second.sample_task_state)
    assert first.task_events == second.task_events
    assert first.generation_parameters == second.generation_parameters


def test_mne_raw_retains_simulation_and_event_provenance() -> None:
    result = generate_event_modulated_synthetic_eeg(
        _simulation(), sampling_frequency_hz=128, n_channels=2, seed=2
    )
    raw = synthetic_participant_to_raw(result, "A")
    metadata = json.loads(raw.info["description"])
    assert metadata["data_source"] == "synthetic"
    assert metadata["simulation_metadata"]["data_source"] == "simulated"
    assert metadata["sample_task_state_count"] == 2
    assert len(metadata["task_events"]) == 3


def test_npz_roundtrip_preserves_event_state_without_pickle(tmp_path) -> None:
    result = generate_event_modulated_synthetic_eeg(
        _simulation(), sampling_frequency_hz=128, n_channels=2, seed=2
    )
    from neurosync.data.synthetic_eeg import save_synthetic_eeg

    target = save_synthetic_eeg(result, tmp_path / "event_synthetic_eeg.npz")
    with np.load(target, allow_pickle=False) as arrays:
        assert set(np.unique(arrays["sample_task_state"])) == {
            "synchronized",
            "desynchronized",
        }
        metadata = json.loads(arrays["generation_parameters_json"].item())
        assert metadata["simulation_metadata"]["data_source"] == "simulated"
        events = metadata["task_events"]
        assert len(events) == 3


def test_rejects_non_simulated_or_incomplete_event_coverage() -> None:
    with pytest.raises(ValueError, match="simulated"):
        generate_event_modulated_synthetic_eeg(_simulation(source="human"))
    with pytest.raises(ValueError, match="cover every"):
        generate_event_modulated_synthetic_eeg(_simulation(gap=True))
