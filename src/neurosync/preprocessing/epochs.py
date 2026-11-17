"""Event-locked epoch construction with explicit boundaries and baselines."""

from __future__ import annotations

import math
from collections.abc import Mapping

import mne
import numpy as np
from numpy.typing import ArrayLike


def epoch_eeg(
    raw: mne.io.BaseRaw,
    events: ArrayLike,
    event_id: Mapping[str, int],
    *,
    tmin: float,
    tmax: float,
    baseline: tuple[float | None, float | None] | None = None,
) -> mne.Epochs:
    """Create MNE epochs from explicit sample-index events.

    ``events`` must use MNE's three-column format: absolute sample index,
    previous event value, and positive event code. Baseline correction defaults
    to ``None``; selecting one is an analysis decision, not an implicit step.
    Events whose complete requested window falls outside the recording are
    rejected with ``ValueError`` instead of silently dropped.
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("raw must be an MNE Raw instance")
    if isinstance(tmin, bool) or not math.isfinite(tmin):
        raise ValueError("tmin must be finite")
    if isinstance(tmax, bool) or not math.isfinite(tmax):
        raise ValueError("tmax must be finite")
    if tmin > tmax:
        raise ValueError("tmin must be less than or equal to tmax")

    event_array = np.asarray(events)
    if event_array.ndim != 2 or event_array.shape[1] != 3:
        raise ValueError("events must have shape (n_events, 3) in MNE event format")
    if event_array.shape[0] == 0 or not np.issubdtype(event_array.dtype, np.integer):
        raise ValueError("events must be a non-empty integer array")
    if not event_id or any(not isinstance(name, str) or not name for name in event_id):
        raise ValueError("event_id must map non-empty event names to integer codes")
    if any(
        isinstance(code, bool) or not isinstance(code, (int, np.integer)) or code <= 0
        for code in event_id.values()
    ):
        raise ValueError("event_id codes must be positive integers")
    if not set(np.unique(event_array[:, 2])).issubset(set(event_id.values())):
        raise ValueError("every event code must be present in event_id")

    sfreq = float(raw.info["sfreq"])
    first_sample = int(raw.first_samp)
    last_sample = first_sample + raw.n_times - 1
    start_offsets = np.rint(event_array[:, 0] - first_sample + tmin * sfreq).astype(int)
    end_offsets = np.rint(event_array[:, 0] - first_sample + tmax * sfreq).astype(int)
    if np.any(start_offsets < 0) or np.any(end_offsets > raw.n_times - 1):
        raise ValueError(
            "each event epoch window must fit completely inside the recording"
        )
    if np.any(event_array[:, 0] < first_sample) or np.any(
        event_array[:, 0] > last_sample
    ):
        raise ValueError("event sample indices must fall inside the recording")

    return mne.Epochs(
        raw.copy(),
        events=event_array.copy(),
        event_id=dict(event_id),
        tmin=float(tmin),
        tmax=float(tmax),
        baseline=baseline,
        preload=True,
        reject_by_annotation=False,
        verbose="ERROR",
    )
