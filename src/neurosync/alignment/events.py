"""Align scripted simulation events to explicitly synthetic MNE recordings."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import Any

import mne
import numpy as np
import pandas as pd

from neurosync.simulation.task import CollaborativeTaskResult, TaskEvent

SIMULATION_EVENT_PREFIX = "NEUROSYNC_SIM_TASK"


def _raw_provenance(raw: mne.io.BaseRaw) -> dict[str, Any]:
    description = raw.info.get("description")
    try:
        metadata = json.loads(description) if isinstance(description, str) else {}
    except json.JSONDecodeError as exc:
        raise ValueError(
            "EEG Raw description must contain valid provenance JSON"
        ) from exc
    if metadata.get("data_source") != "synthetic":
        raise ValueError("EEG recording must be explicitly labeled synthetic")
    if metadata.get("signal_unit") != "V":
        raise ValueError("EEG recording must declare signal_unit as 'V'")
    return metadata


def _simulation_events(
    result: CollaborativeTaskResult,
) -> tuple[TaskEvent, ...]:
    if result.metadata.get("data_source") != "simulated":
        raise ValueError(
            "behavioral events must come from explicitly labeled simulation"
        )
    if not result.events:
        raise ValueError("simulation result contains no task events")
    previous_end = -math.inf
    for event in result.events:
        if event.source != "scripted_simulation_ground_truth":
            raise ValueError(
                "task event source must be scripted simulation ground truth"
            )
        if (
            not math.isfinite(event.start_time_seconds)
            or not math.isfinite(event.end_time_seconds)
            or event.start_time_seconds < 0
            or event.end_time_seconds <= event.start_time_seconds
        ):
            raise ValueError(
                "task event intervals must be finite, positive, and non-negative"
            )
        if event.start_time_seconds < previous_end:
            raise ValueError("task events must be ordered and non-overlapping")
        previous_end = event.end_time_seconds
    return result.events


def annotate_raw_with_task_events(
    raw: mne.io.BaseRaw,
    result: CollaborativeTaskResult,
    *,
    clock_offset_seconds: float = 0.0,
) -> mne.io.BaseRaw:
    """Return a Raw copy with simulated task-state intervals as MNE annotations.

    Behavior and EEG clocks are assumed to share a time base except for the
    caller-supplied offset. Annotation times are in seconds and are mapped by
    MNE to the recording's sample grid when event samples are requested.
    """
    eeg_metadata = _raw_provenance(raw)
    events = _simulation_events(result)
    if not math.isfinite(clock_offset_seconds):
        raise ValueError("clock_offset_seconds must be finite")
    recording_start = float(raw.times[0])
    recording_end = float(raw.times[-1])
    offset_events = [
        (
            event,
            event.start_time_seconds + clock_offset_seconds,
            event.end_time_seconds + clock_offset_seconds,
        )
        for event in events
    ]
    sample_period = 1.0 / float(raw.info["sfreq"])
    start_tolerance = 0.5 * sample_period
    # Annotation intervals end at a boundary after the final included sample;
    # therefore they may end up to one sample period after raw.times[-1].
    end_tolerance = sample_period + 1e-9
    if any(
        start < recording_start - start_tolerance or end > recording_end + end_tolerance
        for _, start, end in offset_events
    ):
        raise ValueError(
            "offset task event intervals must fit within the EEG recording"
        )
    annotations = mne.Annotations(
        onset=[start for _, start, _ in offset_events],
        duration=[end - start for _, start, end in offset_events],
        description=[
            f"{SIMULATION_EVENT_PREFIX}/{event.task_state}"
            for event, _, _ in offset_events
        ],
        orig_time=None,
    )
    aligned = raw.copy()
    aligned.set_annotations(aligned.annotations + annotations)
    updated_metadata = dict(eeg_metadata)
    updated_metadata["task_event_alignment"] = {
        "data_source": "simulated",
        "clock_offset_seconds": float(clock_offset_seconds),
        "time_unit": "s",
        "event_count": len(events),
        "event_time_convention": "half-open [start, end); MNE annotation duration follows this interval",
        "event_state_source": "scripted_simulation_ground_truth",
        "simulation_seed": result.metadata.get("seed"),
        "events": [asdict(event) for event in events],
    }
    aligned.info["description"] = json.dumps(updated_metadata, sort_keys=True)
    return aligned


def epoch_task_events(
    raw: mne.io.BaseRaw,
    result: CollaborativeTaskResult,
    *,
    tmin: float = 0.0,
    tmax: float = 1.0,
    baseline: tuple[float, float] | None = None,
    clock_offset_seconds: float = 0.0,
) -> mne.Epochs:
    """Create MNE Epochs at scripted task-state event onsets.

    Event identifiers are task-state names. Epoch metadata records phase and
    simulation event ID for traceability; no baseline correction is implicit.
    """
    if not math.isfinite(tmin) or not math.isfinite(tmax) or tmin >= tmax:
        raise ValueError("tmin and tmax must be finite with tmin < tmax")
    events = _simulation_events(result)
    annotated = annotate_raw_with_task_events(
        raw, result, clock_offset_seconds=clock_offset_seconds
    )
    sfreq = float(raw.info["sfreq"])
    raw_end = float(raw.times[-1])
    event_id = {
        state: index + 1
        for index, state in enumerate(sorted({event.task_state for event in events}))
    }
    mne_event_id = {
        f"{SIMULATION_EVENT_PREFIX}/{state}": code for state, code in event_id.items()
    }
    event_array, _ = mne.events_from_annotations(
        annotated,
        event_id=mne_event_id,
        use_rounding=True,
        verbose="ERROR",
    )
    if len(event_array) != len(events):
        raise ValueError("MNE annotation events could not be mapped one-to-one")
    aligned_onsets = np.asarray(
        [event.start_time_seconds + clock_offset_seconds for event in events]
    )
    if np.any(aligned_onsets + tmin < -0.5 / sfreq) or np.any(
        aligned_onsets + tmax > raw_end + 0.5 / sfreq
    ):
        raise ValueError("requested epoch windows must fit within the EEG recording")
    metadata = pd.DataFrame(
        {
            "simulation_event_id": [event.event_id for event in events],
            "phase_name": [event.phase_name for event in events],
            "task_state": [event.task_state for event in events],
            "event_time_seconds": aligned_onsets,
            "event_source": [event.source for event in events],
        }
    )
    epochs = mne.Epochs(
        annotated,
        event_array,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        reject_by_annotation=False,
        metadata=metadata,
        on_missing="raise",
        verbose="ERROR",
    )
    if len(epochs) != len(events):
        raise ValueError(
            "one or more task-event epochs were dropped during construction"
        )
    return epochs
