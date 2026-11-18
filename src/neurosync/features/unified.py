"""Event-level EEG and behavior features with explicit source provenance."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import mne
import numpy as np
import pandas as pd

from neurosync.collaboration.synchrony import (
    SynchronyConfig,
    compute_synthetic_coherence,
)
from neurosync.data.synthetic_eeg import SyntheticEEGData
from neurosync.features.spectral import (
    SpectralConfig,
    extract_spectral_features,
)
from neurosync.simulation.task import CollaborativeTaskResult


@dataclass(frozen=True)
class UnifiedFeatureTable:
    """One feature row per scripted task event, retaining source metadata."""

    rows: pd.DataFrame
    metadata: dict[str, Any]


def _event_raw(
    signal: np.ndarray,
    eeg: SyntheticEEGData,
    simulation: CollaborativeTaskResult,
    event_id: int,
) -> mne.io.RawArray:
    info = mne.create_info(
        list(eeg.channel_names), eeg.sampling_frequency_hz, ch_types="eeg"
    )
    info["description"] = json.dumps(
        {
            "data_source": "synthetic",
            "signal_unit": "V",
            "condition": eeg.condition.value,
            "random_seed": eeg.random_seed,
            "participant_label": "event_window",
            "simulation_metadata": simulation.metadata,
            "feature_event_id": event_id,
        },
        sort_keys=True,
    )
    return mne.io.RawArray(signal, info, verbose="ERROR")


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=0))


def extract_unified_features(
    eeg: SyntheticEEGData,
    simulation: CollaborativeTaskResult,
    *,
    session_id: str | None = None,
    spectral_config: SpectralConfig | None = None,
    coherence_config: SynchronyConfig | None = None,
) -> UnifiedFeatureTable:
    """Extract aligned spectral, coherence, and kinematic features per event.

    Each output row spans one half-open simulation event interval ``[start,
    end)``. Signal and behavior clocks are assumed to share a time base; their
    temporal coverage must match the declared simulation duration. Task state
    labels are authored simulation ground truth, not inferred from signals.
    EEG power units and Welch settings follow :func:`extract_spectral_features`;
    coherence remains a descriptive association statistic.
    """
    if eeg.data_source != "synthetic":
        raise ValueError("EEG features require data explicitly labeled synthetic")
    if simulation.metadata.get("data_source") != "simulated":
        raise ValueError("behavior features require explicitly simulated telemetry")
    if not simulation.events:
        raise ValueError("simulation must include at least one labeled event")
    if not isinstance(eeg.channel_names, tuple) or not eeg.channel_names:
        raise ValueError("EEG channel names must be a non-empty tuple")
    expected_shape = (len(eeg.channel_names), eeg.time_seconds.size)
    if (
        eeg.participant_a.shape != expected_shape
        or eeg.participant_b.shape != expected_shape
    ):
        raise ValueError("both participant signals must match channel/time dimensions")
    if eeg.time_seconds.ndim != 1 or eeg.time_seconds.size < 2:
        raise ValueError("EEG time vector must contain at least two samples")
    if not np.isfinite(eeg.time_seconds).all() or not np.all(
        np.diff(eeg.time_seconds) > 0
    ):
        raise ValueError("EEG time vector must be finite and strictly increasing")
    if not np.allclose(np.diff(eeg.time_seconds), 1.0 / eeg.sampling_frequency_hz):
        raise ValueError("EEG time steps must match the declared sampling frequency")
    if (
        not np.isfinite(eeg.participant_a).all()
        or not np.isfinite(eeg.participant_b).all()
    ):
        raise ValueError("EEG signals must contain only finite values")
    if not math.isfinite(eeg.sampling_frequency_hz) or eeg.sampling_frequency_hz <= 0:
        raise ValueError("EEG sampling frequency must be finite and positive")
    if not math.isclose(float(eeg.time_seconds[0]), 0.0, abs_tol=1e-9):
        raise ValueError("EEG time vector must start at zero")
    eeg_dt = 1.0 / eeg.sampling_frequency_hz
    expected_duration = float(simulation.metadata.get("duration_seconds", 0.0))
    eeg_duration = float(eeg.time_seconds[-1] + eeg_dt)
    if expected_duration <= 0 or not math.isclose(
        eeg_duration, expected_duration, rel_tol=0.0, abs_tol=eeg_dt + 1e-9
    ):
        raise ValueError("EEG and simulation durations must describe the same run")
    sim_times = np.asarray(simulation.time_seconds, dtype=float)
    if (
        sim_times.ndim != 1
        or sim_times.size < 2
        or not np.isfinite(sim_times).all()
        or not np.all(np.diff(sim_times) > 0)
    ):
        raise ValueError(
            "simulation time vector must be finite and strictly increasing"
        )
    if not math.isclose(float(sim_times[0]), 0.0, abs_tol=1e-9):
        raise ValueError("simulation time vector must start at zero")
    telemetry = (
        simulation.agent_a_position_m,
        simulation.agent_b_position_m,
        simulation.agent_a_velocity_m_s,
        simulation.agent_b_velocity_m_s,
        simulation.agent_a_speed_m_s,
        simulation.agent_b_speed_m_s,
        simulation.inter_agent_distance_m,
    )
    if any(np.asarray(array).shape[0] != sim_times.size for array in telemetry):
        raise ValueError("all behavioral telemetry must align to simulation time")
    if any(
        np.asarray(array).shape != (sim_times.size, 3)
        for array in (
            simulation.agent_a_position_m,
            simulation.agent_b_position_m,
            simulation.agent_a_velocity_m_s,
            simulation.agent_b_velocity_m_s,
        )
    ):
        raise ValueError(
            "agent position and velocity arrays must have shape (samples, 3)"
        )
    if any(
        np.asarray(array).shape != (sim_times.size,)
        for array in (
            simulation.agent_a_speed_m_s,
            simulation.agent_b_speed_m_s,
            simulation.inter_agent_distance_m,
        )
    ):
        raise ValueError("speed and distance arrays must have shape (samples,)")
    if any(not np.isfinite(array).all() for array in telemetry):
        raise ValueError("behavioral telemetry must contain only finite values")

    spectral = spectral_config or SpectralConfig(window_seconds=2.0)
    coherence = coherence_config or SynchronyConfig(window_seconds=2.0)
    eeg_masked = np.zeros(eeg.time_seconds.size, dtype=bool)
    feature_rows: list[dict[str, Any]] = []
    previous_event_end = 0.0
    for event in simulation.events:
        if event.source != "scripted_simulation_ground_truth":
            raise ValueError(
                "event labels must come from the scripted simulation schedule"
            )
        if (
            not math.isfinite(event.start_time_seconds)
            or not math.isfinite(event.end_time_seconds)
            or event.start_time_seconds < previous_event_end
            or event.end_time_seconds <= event.start_time_seconds
            or event.end_time_seconds > expected_duration + 1e-9
        ):
            raise ValueError(
                "events must be ordered, non-overlapping, and within the run"
            )
        previous_event_end = event.end_time_seconds
        eeg_mask = (eeg.time_seconds >= event.start_time_seconds) & (
            eeg.time_seconds < event.end_time_seconds
        )
        behavior_mask = (sim_times >= event.start_time_seconds) & (
            sim_times < event.end_time_seconds
        )
        if np.count_nonzero(eeg_mask) < 4 or np.count_nonzero(behavior_mask) < 2:
            raise ValueError("each event must contain enough EEG and behavior samples")
        eeg_masked |= eeg_mask
        duration = event.end_time_seconds - event.start_time_seconds
        if spectral.window_seconds > duration or coherence.window_seconds > duration:
            raise ValueError("spectral/coherence windows must fit each event interval")

        raw_a = _event_raw(
            eeg.participant_a[:, eeg_mask], eeg, simulation, event.event_id
        )
        raw_b = _event_raw(
            eeg.participant_b[:, eeg_mask], eeg, simulation, event.event_id
        )
        features_a = extract_spectral_features(raw_a, spectral)
        features_b = extract_spectral_features(raw_b, spectral)
        coherence_result = compute_synthetic_coherence(raw_a, raw_b, coherence)

        bmask = behavior_mask
        speed_a = np.asarray(simulation.agent_a_speed_m_s)[bmask]
        speed_b = np.asarray(simulation.agent_b_speed_m_s)[bmask]
        distance = np.asarray(simulation.inter_agent_distance_m)[bmask]
        mean_speed_a, std_speed_a = _mean_std(speed_a)
        mean_speed_b, std_speed_b = _mean_std(speed_b)
        corr = 0.0
        if np.std(speed_a) > 0 and np.std(speed_b) > 0:
            corr = float(np.corrcoef(speed_a, speed_b)[0, 1])
            if not math.isfinite(corr):
                corr = 0.0
        positions_a = np.asarray(simulation.agent_a_position_m)[bmask]
        positions_b = np.asarray(simulation.agent_b_position_m)[bmask]
        velocity_a = np.asarray(simulation.agent_a_velocity_m_s)[bmask]
        velocity_b = np.asarray(simulation.agent_b_velocity_m_s)[bmask]
        row: dict[str, Any] = {
            "session_id": session_id
            or f"sim-seed-{simulation.metadata.get('seed', 'unknown')}",
            "simulation_seed": simulation.metadata.get("seed"),
            "eeg_seed": eeg.random_seed,
            "event_id": event.event_id,
            "phase_name": event.phase_name,
            "task_state": event.task_state,
            "label_source": event.source,
            "event_start_seconds": event.start_time_seconds,
            "event_end_seconds": event.end_time_seconds,
            "event_duration_seconds": duration,
            "data_source_eeg": "synthetic",
            "data_source_behavior": "simulated",
            "velocity_magnitude_correlation": corr,
            "mean_inter_agent_distance_m": float(np.mean(distance)),
            "std_inter_agent_distance_m": float(np.std(distance, ddof=0)),
            "minimum_inter_agent_distance_m": float(np.min(distance)),
            "agent_a_path_length_m": float(
                np.linalg.norm(np.diff(positions_a, axis=0), axis=1).sum()
            ),
            "agent_b_path_length_m": float(
                np.linalg.norm(np.diff(positions_b, axis=0), axis=1).sum()
            ),
            "agent_a_mean_speed_m_s": mean_speed_a,
            "agent_a_std_speed_m_s": std_speed_a,
            "agent_a_max_speed_m_s": float(np.max(speed_a)),
            "agent_b_mean_speed_m_s": mean_speed_b,
            "agent_b_std_speed_m_s": std_speed_b,
            "agent_b_max_speed_m_s": float(np.max(speed_b)),
            "agent_a_mean_velocity_x_m_s": float(np.mean(velocity_a[:, 0])),
            "agent_b_mean_velocity_x_m_s": float(np.mean(velocity_b[:, 0])),
            "eeg_sampling_frequency_hz": eeg.sampling_frequency_hz,
            "spectral_frequency_resolution_hz": features_a.metadata[
                "frequency_resolution_hz"
            ],
        }
        for participant, features in (("a", features_a), ("b", features_b)):
            for key, value in features.channel_summary.items():
                if key != "channel_count":
                    row[f"eeg_{participant}_{key}"] = value
        for band, value in coherence_result.band_mean_coherence.items():
            row[f"eeg_mean_{band}_coherence"] = value
        feature_rows.append(row)

    if not eeg_masked.all():
        raise ValueError("task events must cover every EEG sample without gaps")
    rows = pd.DataFrame(feature_rows)
    values = rows.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("unified numeric features must all be finite")
    metadata = {
        "data_source_eeg": "synthetic",
        "data_source_behavior": "simulated",
        "simulation_engine": simulation.metadata.get("engine"),
        "simulation_seed": simulation.metadata.get("seed"),
        "eeg_seed": eeg.random_seed,
        "signal_unit": "V",
        "behavior_position_unit": "m",
        "behavior_velocity_unit": "m/s",
        "row_unit": "one scripted task event interval",
        "interval_convention": "half-open [start, end)",
        "clock_assumption": "shared time base; no offset applied by this extractor",
        "label_source": "scripted_simulation_ground_truth",
        "coherence_interpretation": "descriptive statistical association; no synchrony inference",
        "limitations": [
            "All labels and signals are constructed/simulated, not human measurements.",
            "Features do not establish collaboration quality or real-world validity.",
            "Feature rows within a simulation run are dependent; group validation by run/seed is required.",
        ],
    }
    return UnifiedFeatureTable(rows=rows, metadata=metadata)
