"""Visual regression contracts without image snapshots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from neurosync.collaboration import compute_synthetic_coherence
from neurosync.data.synthetic_eeg import generate_synthetic_eeg
from neurosync.features.spectral import extract_spectral_features
from neurosync.preprocessing.raw import synthetic_participant_to_raw
from neurosync.visualization import (
    plot_coherence,
    plot_spectral_features,
    plot_synthetic_signals,
)


@pytest.fixture
def processed_pair():
    data = generate_synthetic_eeg(
        sampling_frequency_hz=256,
        duration_seconds=8,
        n_channels=2,
        seed=81,
    )
    return (
        data,
        synthetic_participant_to_raw(data, "A"),
        synthetic_participant_to_raw(data, "B"),
    )


def test_signal_plot_identifies_synthetic_source_and_units(processed_pair) -> None:
    data, _, _ = processed_pair
    figure = plot_synthetic_signals(data, participant="B", channels=(0, 1))
    assert len(figure.axes) == 2
    assert "SYNTHETIC EEG" in figure._suptitle.get_text()
    assert "µV" in figure.axes[0].get_ylabel()
    plt.close(figure)


def test_spectral_plot_contains_psd_units(processed_pair) -> None:
    _, raw_a, _ = processed_pair
    figure = plot_spectral_features(extract_spectral_features(raw_a))
    assert len(figure.axes) == 2
    assert "V²/Hz" in figure.axes[1].get_ylabel()
    assert "SYNTHETIC EEG" in figure._suptitle.get_text()
    plt.close(figure)


def test_coherence_plot_uses_cautious_language(processed_pair) -> None:
    _, raw_a, raw_b = processed_pair
    figure = plot_coherence(compute_synthetic_coherence(raw_a, raw_b))
    assert len(figure.axes) == 2
    assert "not proof of synchrony" in figure._suptitle.get_text()
    plt.close(figure)


def test_signal_plot_rejects_invalid_selection(processed_pair) -> None:
    data, _, _ = processed_pair
    with pytest.raises(ValueError, match="participant"):
        plot_synthetic_signals(data, participant="C")
    with pytest.raises(ValueError, match="channels"):
        plot_synthetic_signals(data, channels=(2,))
