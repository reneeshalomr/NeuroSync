"""Plots for descriptive synthetic spectral features and coherence."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from neurosync.collaboration.synchrony import SynchronyResult
from neurosync.features.spectral import SpectralFeatureResult


def plot_spectral_features(result: SpectralFeatureResult) -> Figure:
    """Plot channel relative band power and PSD, labeled as synthetic data."""
    if result.metadata.get("data_source") != "synthetic":
        raise ValueError("plot_spectral_features requires synthetic provenance")
    names = [item.channel_name for item in result.channel_features]
    bands = list(result.channel_features[0].relative_band_power)
    power = np.array(
        [
            [item.relative_band_power[band] for band in bands]
            for item in result.channel_features
        ]
    )
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    axes[0].bar(np.arange(len(bands)), power.mean(axis=0), color="#3178a8")
    axes[0].set_xticks(np.arange(len(bands)), bands)
    axes[0].set_ylabel("Mean relative band power")
    axes[0].set_ylim(bottom=0)
    axes[0].grid(axis="y", alpha=0.25)
    for index, name in enumerate(names):
        axes[1].semilogy(
            result.frequencies_hz,
            result.psd_v2_per_hz[index],
            label=name,
        )
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD (V²/Hz)")
    axes[1].legend(title="Synthetic channels")
    axes[1].grid(True, alpha=0.25)
    figure.suptitle("SYNTHETIC EEG · Descriptive spectral features")
    return figure


def plot_coherence(result: SynchronyResult) -> Figure:
    """Plot per-channel magnitude-squared coherence and band summaries."""
    if result.metadata.get("data_source") != "synthetic":
        raise ValueError("plot_coherence requires synthetic provenance")
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    for name, values in zip(result.channel_names, result.coherence, strict=True):
        axes[0].plot(result.frequencies_hz, values, label=name)
    axes[0].set_ylabel("Magnitude-squared coherence")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(title="Matched channels")
    axes[0].grid(True, alpha=0.25)
    bands = list(result.band_mean_coherence)
    axes[1].bar(bands, [result.band_mean_coherence[band] for band in bands])
    axes[1].set_ylabel("Mean coherence (descriptive)")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("SYNTHETIC EEG · Cross-signal association; not proof of synchrony")
    return figure
