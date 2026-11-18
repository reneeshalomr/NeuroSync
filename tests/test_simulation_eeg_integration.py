"""End-to-end verification from real PyBullet events into synthetic EEG/MNE."""

from __future__ import annotations

import json

import numpy as np
import pytest

from neurosync.alignment import annotate_raw_with_task_events, epoch_task_events
from neurosync.data.event_driven import generate_event_modulated_synthetic_eeg
from neurosync.preprocessing.raw import synthetic_participant_to_raw
from neurosync.simulation import CollaborativeTaskConfig, run_collaborative_task
from neurosync.simulation import task as simulation_task


@pytest.mark.skipif(
    simulation_task.bullet is None,
    reason="integration test requires the optional real PyBullet engine",
)
def test_real_pybullet_events_generate_and_align_synthetic_eeg() -> None:
    simulation = run_collaborative_task(
        CollaborativeTaskConfig(sampling_frequency_hz=20.0, seed=2026)
    )
    eeg = generate_event_modulated_synthetic_eeg(
        simulation,
        sampling_frequency_hz=128.0,
        n_channels=4,
        seed=72,
    )
    raw = synthetic_participant_to_raw(eeg, "A")

    aligned = annotate_raw_with_task_events(raw, simulation)
    epochs = epoch_task_events(raw, simulation, tmin=0.0, tmax=0.25)
    raw_metadata = json.loads(aligned.info["description"])
    epoch_states = list(epochs.metadata["task_state"])

    assert simulation.metadata["engine"] == "PyBullet DIRECT"
    assert eeg.data_source == "synthetic"
    assert eeg.simulation_metadata["data_source"] == "simulated"
    assert len(aligned.annotations) == len(simulation.events)
    np.testing.assert_allclose(
        aligned.annotations.onset,
        [event.start_time_seconds for event in simulation.events],
    )
    assert len(epochs) == len(simulation.events)
    assert epoch_states == [event.task_state for event in simulation.events]
    assert raw_metadata["task_event_alignment"]["simulation_seed"] == 2026
