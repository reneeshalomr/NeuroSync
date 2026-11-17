"""Tests for MNE mapping of scripted simulation event timestamps."""

from __future__ import annotations

import json

import mne
import numpy as np
import pytest

from neurosync.alignment import annotate_raw_with_task_events, epoch_task_events
from neurosync.simulation.task import CollaborativeTaskResult, TaskEvent


@pytest.fixture
def raw_and_task_events():
    sfreq = 128.0
    samples = int(7 * sfreq)
    rng = np.random.default_rng(21)
    info = mne.create_info(["EEG001", "EEG002"], sfreq, ch_types="eeg")
    info["description"] = json.dumps(
        {"data_source": "synthetic", "signal_unit": "V", "condition": "fixture"}
    )
    raw = mne.io.RawArray(rng.normal(size=(2, samples)) * 1e-6, info, verbose="ERROR")
    events = (
        TaskEvent(0, "approach", "synchronized", 0.0, 2.0),
        TaskEvent(1, "handover", "desynchronized", 2.0, 4.0),
        TaskEvent(2, "retreat", "synchronized", 4.0, 6.0),
    )
    result = CollaborativeTaskResult(
        time_seconds=np.arange(samples) / sfreq,
        agent_a_position_m=np.zeros((samples, 3)),
        agent_b_position_m=np.zeros((samples, 3)),
        agent_a_velocity_m_s=np.zeros((samples, 3)),
        agent_b_velocity_m_s=np.zeros((samples, 3)),
        agent_a_speed_m_s=np.zeros(samples),
        agent_b_speed_m_s=np.zeros(samples),
        inter_agent_distance_m=np.zeros(samples),
        events=events,
        velocity_correlation=0.0,
        idle_duration_seconds={"agent_a": 6.0, "agent_b": 6.0},
        metadata={"data_source": "simulated", "seed": 21},
    )
    return raw, result


def test_annotations_preserve_synthetic_raw_and_map_event_times(
    raw_and_task_events,
) -> None:
    raw, result = raw_and_task_events
    aligned = annotate_raw_with_task_events(raw, result)
    assert len(raw.annotations) == 0
    assert len(aligned.annotations) == 3
    assert list(aligned.annotations.description) == [
        "NEUROSYNC_SIM_TASK/synchronized",
        "NEUROSYNC_SIM_TASK/desynchronized",
        "NEUROSYNC_SIM_TASK/synchronized",
    ]
    np.testing.assert_allclose(aligned.annotations.onset, [0.0, 2.0, 4.0])
    metadata = json.loads(aligned.info["description"])
    assert metadata["data_source"] == "synthetic"
    assert metadata["task_event_alignment"]["data_source"] == "simulated"


def test_clock_offset_maps_to_raw_annotations(raw_and_task_events) -> None:
    raw, result = raw_and_task_events
    aligned = annotate_raw_with_task_events(raw, result, clock_offset_seconds=0.25)
    np.testing.assert_allclose(aligned.annotations.onset, [0.25, 2.25, 4.25])


def test_epochs_carry_state_codes_and_event_metadata(raw_and_task_events) -> None:
    raw, result = raw_and_task_events
    epochs = epoch_task_events(raw, result, tmin=0.0, tmax=0.25)
    assert len(epochs) == 3
    assert epochs.event_id == {"desynchronized": 1, "synchronized": 2}
    assert epochs.metadata is not None
    assert list(epochs.metadata["phase_name"]) == ["approach", "handover", "retreat"]
    assert (
        list(epochs.metadata["event_source"])
        == ["scripted_simulation_ground_truth"] * 3
    )
    assert epochs.baseline is None


def test_rejects_non_synthetic_or_non_simulated_provenance(raw_and_task_events) -> None:
    raw, result = raw_and_task_events
    raw.info["description"] = json.dumps({"data_source": "human", "signal_unit": "V"})
    with pytest.raises(ValueError, match="synthetic"):
        annotate_raw_with_task_events(raw, result)
    raw.info["description"] = json.dumps(
        {"data_source": "synthetic", "signal_unit": "V"}
    )
    result.metadata["data_source"] = "simulated"
    raw, result = raw_and_task_events
    result.metadata["data_source"] = "human"
    with pytest.raises(ValueError, match="simulation"):
        annotate_raw_with_task_events(raw, result)


def test_rejects_out_of_range_offset_and_epoch_window(raw_and_task_events) -> None:
    raw, result = raw_and_task_events
    with pytest.raises(ValueError, match="fit within"):
        annotate_raw_with_task_events(raw, result, clock_offset_seconds=1.5)
    with pytest.raises(ValueError, match="epoch windows"):
        epoch_task_events(raw, result, tmin=-0.5, tmax=2.5)
