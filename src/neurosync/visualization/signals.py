"""Plots for synthetic signals with provenance and physical units visible."""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from neurosync.data.synthetic_eeg import SyntheticEEGData


def plot_synthetic_signals(
    data: SyntheticEEGData,
    participant: Literal["A", "B"] = "A",
    channels: tuple[int, ...] | None = None,
    duration_seconds: float = 5.0,
) -> Figure:
    """Plot selected channels for one simulated participant in microvolts."""
    if data.data_source != "synthetic":
        raise ValueError("plot_synthetic_signals accepts only synthetic data")
    if participant not in ("A", "B"):
        raise ValueError("participant must be 'A' or 'B'")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    signal = data.participant_a if participant == "A" else data.participant_b
    selected = tuple(range(signal.shape[0])) if channels is None else channels
    if not selected or any(index < 0 or index >= signal.shape[0] for index in selected):
        raise ValueError("channels must contain valid channel indices")
    sample_stop = min(
        signal.shape[1], int(duration_seconds * data.sampling_frequency_hz)
    )
    if sample_stop < 2:
        raise ValueError("duration_seconds must cover at least two samples")

    figure, axes = plt.subplots(len(selected), 1, sharex=True, squeeze=False)
    for axis, index in zip(axes[:, 0], selected, strict=True):
        axis.plot(data.time_seconds[:sample_stop], signal[index, :sample_stop] * 1e6)
        axis.set_ylabel(f"{data.channel_names[index]}\n(µV)")
        axis.grid(True, alpha=0.25)
    axes[-1, 0].set_xlabel("Time (s)")
    figure.suptitle(
        f"SYNTHETIC EEG · Participant {participant} · {data.condition.value} · seed={data.random_seed}"
    )
    figure.tight_layout()
    return figure
