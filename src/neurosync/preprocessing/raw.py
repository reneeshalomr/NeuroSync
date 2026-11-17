"""Conversion of a synthetic participant signal to an MNE Raw object."""

from __future__ import annotations

import json
from typing import Literal

import mne
import numpy as np
import scipy

from neurosync.data.synthetic_eeg import SyntheticEEGData


def synthetic_participant_to_raw(
    data: SyntheticEEGData,
    participant: Literal["A", "B"] = "A",
) -> mne.io.RawArray:
    """Create an MNE RawArray for one simulated participant.

    The paired participants remain separate MNE recordings. Data are already
    expressed in volts by the generator; no unit conversion is applied here.
    Provenance is stored in ``raw.info['description']``. This entry point
    intentionally accepts only the project's synthetic data contract.
    """
    if participant not in ("A", "B"):
        raise ValueError("participant must be 'A' or 'B'")
    if data.data_source != "synthetic":
        raise ValueError("this converter accepts only data labeled synthetic")
    if data.generation_parameters.get("signal_unit") != "V":
        raise ValueError("synthetic EEG signals must declare units as volts ('V')")

    signal = data.participant_a if participant == "A" else data.participant_b
    if signal.ndim != 2 or signal.shape[0] != len(data.channel_names):
        raise ValueError(
            "signal shape must be (channels, samples) and match channel_names"
        )
    if signal.shape[1] != data.time_seconds.size or signal.shape[1] < 2:
        raise ValueError(
            "signal sample count must match the time vector and contain at least 2 samples"
        )
    if not np.isfinite(signal).all() or not np.isfinite(data.time_seconds).all():
        raise ValueError("signal and time vector must contain only finite values")
    if not np.isfinite(data.sampling_frequency_hz) or data.sampling_frequency_hz <= 0:
        raise ValueError("sampling_frequency_hz must be finite and positive")
    expected_time = (
        np.arange(signal.shape[1], dtype=np.float64) / data.sampling_frequency_hz
    )
    if not np.allclose(data.time_seconds, expected_time, rtol=1e-10, atol=1e-12):
        raise ValueError(
            "time vector must start at zero and match the sampling frequency"
        )
    if len(set(data.channel_names)) != len(data.channel_names):
        raise ValueError("channel names must be unique")

    info = mne.create_info(
        ch_names=list(data.channel_names),
        sfreq=float(data.sampling_frequency_hz),
        ch_types=["eeg"] * len(data.channel_names),
    )
    provenance = {
        "data_source": "synthetic",
        "condition": data.condition.value,
        "random_seed": data.random_seed,
        "participant_label": f"synthetic_{participant}",
        "signal_unit": "V",
        "mne_version": mne.__version__,
        "scipy_version": scipy.__version__,
        "generation_parameters": data.generation_parameters,
    }
    info["description"] = json.dumps(provenance, sort_keys=True)
    return mne.io.RawArray(signal.copy(), info, verbose="ERROR")
