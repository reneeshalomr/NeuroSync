"""Tests for MNE conversion, preprocessing, and event-locked epoching."""

import json

import mne
import numpy as np
import pytest

from neurosync.data.synthetic_eeg import generate_synthetic_eeg
from neurosync.preprocessing import (
    PreprocessingConfig,
    epoch_eeg,
    filter_eeg,
    synthetic_participant_to_raw,
)


@pytest.fixture
def synthetic_raw() -> mne.io.RawArray:
    data = generate_synthetic_eeg(
        sampling_frequency_hz=256,
        duration_seconds=10,
        n_channels=4,
        seed=2026,
    )
    return synthetic_participant_to_raw(data, participant="A")


def test_raw_conversion_preserves_data_units_and_provenance() -> None:
    data = generate_synthetic_eeg(duration_seconds=2, n_channels=3, seed=18)
    raw = synthetic_participant_to_raw(data, participant="B")
    np.testing.assert_array_equal(raw.get_data(), data.participant_b)
    assert raw.info["sfreq"] == data.sampling_frequency_hz
    assert raw.ch_names == list(data.channel_names)
    assert raw.get_channel_types() == ["eeg"] * data.n_channels
    metadata = json.loads(raw.info["description"])
    assert metadata["data_source"] == "synthetic"
    assert metadata["signal_unit"] == "V"
    assert metadata["participant_label"] == "synthetic_B"
    assert metadata["mne_version"] == mne.__version__
    assert metadata["scipy_version"]


def test_raw_conversion_rejects_invalid_participant() -> None:
    data = generate_synthetic_eeg()
    with pytest.raises(ValueError, match="participant"):
        synthetic_participant_to_raw(data, participant="C")  # type: ignore[arg-type]


def test_filter_returns_referenced_copy_and_records_settings(
    synthetic_raw: mne.io.RawArray,
) -> None:
    original = synthetic_raw.get_data().copy()
    config = PreprocessingConfig(
        highpass_hz=1.0,
        lowpass_hz=40.0,
        average_reference=True,
    )
    processed = filter_eeg(synthetic_raw, config)

    assert processed is not synthetic_raw
    np.testing.assert_array_equal(synthetic_raw.get_data(), original)
    assert processed.get_data().shape == original.shape
    assert np.isfinite(processed.get_data()).all()
    assert not np.array_equal(processed.get_data(), original)
    np.testing.assert_allclose(processed.get_data().mean(axis=0), 0.0, atol=1e-12)
    metadata = json.loads(processed.info["description"])
    assert metadata["data_source"] == "synthetic"
    assert metadata["preprocessing"]["highpass_hz"] == 1.0
    assert metadata["preprocessing"]["lowpass_hz"] == 40.0
    assert metadata["preprocessing"]["average_reference"] is True
    assert metadata["preprocessing"]["mne_version"] == mne.__version__
    assert metadata["preprocessing"]["scipy_version"]


def test_filter_without_average_reference_is_supported(
    synthetic_raw: mne.io.RawArray,
) -> None:
    processed = filter_eeg(
        synthetic_raw,
        PreprocessingConfig(average_reference=False),
    )
    metadata = json.loads(processed.info["description"])
    assert metadata["preprocessing"]["average_reference"] is False


def test_optional_notch_filter_is_applied(synthetic_raw: mne.io.RawArray) -> None:
    processed = filter_eeg(
        synthetic_raw,
        PreprocessingConfig(notch_frequencies_hz=(50.0,)),
    )
    metadata = json.loads(processed.info["description"])
    assert metadata["preprocessing"]["notch_frequencies_hz"] == [50.0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"highpass_hz": 40.0, "lowpass_hz": 40.0},
        {"highpass_hz": 41.0, "lowpass_hz": 40.0},
        {"highpass_hz": 0.0},
        {"lowpass_hz": float("nan")},
        {"highpass_hz": None, "lowpass_hz": None},
        {"notch_frequencies_hz": (0.0,)},
    ],
)
def test_invalid_preprocessing_config_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PreprocessingConfig(**kwargs)  # type: ignore[arg-type]


def test_filter_cutoffs_must_be_below_nyquist(synthetic_raw: mne.io.RawArray) -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        filter_eeg(synthetic_raw, PreprocessingConfig(lowpass_hz=128.0))


def test_notch_frequency_must_be_below_nyquist(synthetic_raw: mne.io.RawArray) -> None:
    with pytest.raises(ValueError, match="notch frequencies"):
        filter_eeg(
            synthetic_raw,
            PreprocessingConfig(notch_frequencies_hz=(128.0,)),
        )


def test_epochs_use_explicit_events_and_no_implicit_baseline(
    synthetic_raw: mne.io.RawArray,
) -> None:
    events = np.array(
        [[256, 0, 1], [3 * 256, 0, 2]],
        dtype=np.int64,
    )
    epochs = epoch_eeg(
        synthetic_raw,
        events,
        {"assist": 1, "wait": 2},
        tmin=-0.2,
        tmax=0.5,
    )
    assert len(epochs) == 2
    assert epochs.get_data().shape == (2, 4, 180)
    assert epochs.baseline is None
    assert epochs.event_id == {"assist": 1, "wait": 2}
    np.testing.assert_array_equal(epochs.events, events)


@pytest.mark.parametrize(
    "events,event_id,tmin,tmax",
    [
        (np.array([1, 0, 1]), {"assist": 1}, -0.2, 0.5),
        (np.array([[256.0, 0.0, 1.0]]), {"assist": 1}, -0.2, 0.5),
        (np.array([[256, 0, 3]], dtype=np.int64), {"assist": 1}, -0.2, 0.5),
        (np.array([[10, 0, 1]], dtype=np.int64), {"assist": 1}, -0.2, 0.5),
        (np.array([[256, 0, 1]], dtype=np.int64), {"assist": 1}, 0.5, -0.2),
    ],
)
def test_invalid_epoch_inputs_are_rejected(
    synthetic_raw: mne.io.RawArray,
    events: np.ndarray,
    event_id: dict[str, int],
    tmin: float,
    tmax: float,
) -> None:
    with pytest.raises(ValueError):
        epoch_eeg(synthetic_raw, events, event_id, tmin=tmin, tmax=tmax)
