"""Synthetic EEG with shared-component amplitude driven by task events."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from neurosync.data.synthetic_eeg import (
    ExperimentalCondition,
    SyntheticEEGData,
    generate_synthetic_eeg,
)
from neurosync.simulation.task import CollaborativeTaskResult

SYNCHRONIZED_COUPLING_SCALE = 1.0
DESYNCHRONIZED_COUPLING_SCALE = 0.05


def generate_event_modulated_synthetic_eeg(
    simulation: CollaborativeTaskResult,
    *,
    sampling_frequency_hz: float = 256.0,
    n_channels: int = 8,
    seed: int = 42,
) -> SyntheticEEGData:
    """Generate synthetic EEG-like signals whose shared component follows task events.

    Independent background and channel noise come from the existing LOW shared
    component generator using ``seed``. That generator's residual shared term
    is removed before an event-scaled version is injected. Synchronized task
    states use scale 1.0 and desynchronized states scale 0.05. These values are
    software construction parameters, not neuroscience estimates.
    """
    if simulation.metadata.get("data_source") != "simulated":
        raise ValueError(
            "event-driven EEG requires explicitly simulated task telemetry"
        )
    if not simulation.events:
        raise ValueError("simulation must include task-state events")
    duration = float(simulation.metadata.get("duration_seconds", 0.0))
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError(
            "simulation provenance must declare a positive duration_seconds"
        )
    if simulation.events[0].start_time_seconds != 0.0:
        raise ValueError(
            "task events must begin at time zero to cover the EEG synthesis window"
        )
    if not np.isclose(simulation.events[-1].end_time_seconds, duration, atol=1e-9):
        raise ValueError("task events must cover the complete simulation duration")

    base = generate_synthetic_eeg(
        sampling_frequency_hz=sampling_frequency_hz,
        duration_seconds=duration,
        n_channels=n_channels,
        seed=seed,
        condition=ExperimentalCondition.LOW_SHARED_COMPONENT,
    )
    sample_states = np.full(base.n_samples, "unlabeled", dtype="<U16")
    for event in simulation.events:
        if event.task_state not in {"synchronized", "desynchronized"}:
            raise ValueError(f"unsupported task state {event.task_state!r}")
        if event.source != "scripted_simulation_ground_truth":
            raise ValueError(
                "event states must originate from scripted simulation ground truth"
            )
        event_mask = (base.time_seconds >= event.start_time_seconds) & (
            base.time_seconds < event.end_time_seconds
        )
        sample_states[event_mask] = event.task_state
    if np.any(sample_states == "unlabeled"):
        raise ValueError("task events do not cover every synthetic EEG sample")

    envelope = np.where(
        sample_states == "synchronized",
        SYNCHRONIZED_COUPLING_SCALE,
        DESYNCHRONIZED_COUPLING_SCALE,
    )
    low_scale = float(base.generation_parameters["shared_component_scale"])
    shared = base.shared_component[None, :]
    event_component = shared * (envelope[None, :] / low_scale)
    participant_a = base.participant_a - shared + event_component
    participant_b = base.participant_b - shared + event_component
    event_records = tuple(asdict(event) for event in simulation.events)
    parameters: dict[str, Any] = {
        **base.generation_parameters,
        "generator": "event_driven_synthetic_eeg",
        "base_generator_condition": ExperimentalCondition.LOW_SHARED_COMPONENT.value,
        "task_state_to_shared_component_scale": {
            "synchronized": SYNCHRONIZED_COUPLING_SCALE,
            "desynchronized": DESYNCHRONIZED_COUPLING_SCALE,
        },
        "simulation_seed": simulation.metadata.get("seed"),
        "simulation_engine": simulation.metadata.get("engine"),
        "simulation_event_source": "scripted_simulation_ground_truth",
        "interpretation_limit": (
            "Event-modulated coupling is authored synthetic signal construction; "
            "it does not model or establish human neural synchrony."
        ),
    }
    return SyntheticEEGData(
        participant_a=participant_a,
        participant_b=participant_b,
        shared_component=event_component[0],
        sampling_frequency_hz=base.sampling_frequency_hz,
        channel_names=base.channel_names,
        time_seconds=base.time_seconds,
        condition=ExperimentalCondition.LOW_SHARED_COMPONENT,
        random_seed=seed,
        generation_parameters=parameters,
        data_source="synthetic",
        sample_task_state=sample_states,
        task_events=event_records,
        simulation_metadata=dict(simulation.metadata),
    )
