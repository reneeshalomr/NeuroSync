"""Contract tests for the explicitly synthetic EEG generator."""

import json

import numpy as np
import pytest

from neurosync.data.synthetic_eeg import (
    MAX_BASE_OSCILLATORY_FREQUENCY_HZ,
    MAX_OSCILLATORY_FREQUENCY_HZ,
    ExperimentalCondition,
    generate_synthetic_eeg,
    save_synthetic_eeg,
)


def test_signal_dimensions_and_channel_names() -> None:
    data = generate_synthetic_eeg(duration_seconds=2.0, n_channels=4, seed=7)
    expected_samples = round(2.0 * data.sampling_frequency_hz)
    assert data.participant_a.shape == (4, expected_samples)
    assert data.participant_b.shape == (4, expected_samples)
    assert data.shared_component.shape == (expected_samples,)
    assert data.channel_names == ("EEG001", "EEG002", "EEG003", "EEG004")
    assert data.n_channels == 4
    assert data.n_samples == expected_samples


def test_time_vector_sampling_frequency_and_volt_units() -> None:
    data = generate_synthetic_eeg(
        sampling_frequency_hz=200, duration_seconds=1.25, seed=2
    )
    assert data.sampling_frequency_hz == 200
    assert data.time_seconds[0] == 0.0
    assert data.time_seconds.size == 250
    assert np.allclose(np.diff(data.time_seconds), 1 / 200)
    assert data.time_seconds[-1] < 1.25
    assert data.generation_parameters["signal_unit"] == "V"
    # Default synthetic amplitudes are a few microvolts, represented here in V.
    assert 1e-7 < np.std(data.participant_a) < 1e-4
    assert 1e-7 < np.std(data.participant_b) < 1e-4
    assert np.max(np.abs(data.shared_component)) < 1e-4


@pytest.mark.parametrize("sampling_frequency_hz", [91.0, 128.0, 200.0, 256.0])
def test_oscillatory_components_are_below_nyquist(sampling_frequency_hz: float) -> None:
    data = generate_synthetic_eeg(sampling_frequency_hz=sampling_frequency_hz, seed=13)
    parameters = data.generation_parameters
    nyquist_hz = sampling_frequency_hz / 2
    band_upper_bounds = [
        bounds[1] for bounds in parameters["frequency_bands_hz"].values()
    ]
    assert max(band_upper_bounds) == MAX_BASE_OSCILLATORY_FREQUENCY_HZ
    assert (
        MAX_BASE_OSCILLATORY_FREQUENCY_HZ
        + parameters["amplitude_modulation_frequency_hz"][1]
        == MAX_OSCILLATORY_FREQUENCY_HZ
    )
    assert parameters["maximum_oscillatory_frequency_hz"] < nyquist_hz
    assert parameters["nyquist_frequency_hz"] == nyquist_hz


def test_high_condition_injects_twenty_times_low_shared_component() -> None:
    high = generate_synthetic_eeg(
        seed=31, condition=ExperimentalCondition.HIGH_SHARED_COMPONENT
    )
    low = generate_synthetic_eeg(
        seed=31, condition=ExperimentalCondition.LOW_SHARED_COMPONENT
    )

    # Same seed means the underlying frequencies and phases match. The shared
    # scale is 1.0 vs 0.05, so compare the injected waveform directly instead
    # of relying on a sample correlation that can fluctuate with noise/window.
    np.testing.assert_allclose(
        high.shared_component,
        20 * low.shared_component,
        rtol=1e-12,
        atol=1e-18,
    )
    assert (
        high.generation_parameters["shared_component_scale"]
        > low.generation_parameters["shared_component_scale"]
    )


def test_identical_seed_reproduces_signals_and_metadata() -> None:
    first = generate_synthetic_eeg(
        seed=99, condition=ExperimentalCondition.HIGH_SHARED_COMPONENT
    )
    second = generate_synthetic_eeg(
        seed=99, condition=ExperimentalCondition.HIGH_SHARED_COMPONENT
    )
    np.testing.assert_array_equal(first.participant_a, second.participant_a)
    np.testing.assert_array_equal(first.participant_b, second.participant_b)
    np.testing.assert_array_equal(first.shared_component, second.shared_component)
    np.testing.assert_array_equal(first.time_seconds, second.time_seconds)
    assert first.sampling_frequency_hz == second.sampling_frequency_hz
    assert first.channel_names == second.channel_names
    assert first.condition is second.condition
    assert first.random_seed == second.random_seed
    assert first.generation_parameters == second.generation_parameters
    assert first.data_source == second.data_source == "synthetic"


def test_different_seeds_change_outputs() -> None:
    first = generate_synthetic_eeg(seed=10)
    second = generate_synthetic_eeg(seed=11)
    assert not np.array_equal(first.participant_a, second.participant_a)
    assert not np.array_equal(first.participant_b, second.participant_b)


@pytest.mark.parametrize("condition", list(ExperimentalCondition))
def test_conditions_generate_finite_valid_signals(
    condition: ExperimentalCondition,
) -> None:
    data = generate_synthetic_eeg(condition=condition, seed=3)
    assert data.condition is condition
    assert data.data_source == "synthetic"
    assert data.participant_a.shape == data.participant_b.shape
    assert np.isfinite(data.participant_a).all()
    assert np.isfinite(data.participant_b).all()
    assert np.isfinite(data.shared_component).all()
    assert np.isfinite(data.time_seconds).all()
    assert np.std(data.participant_a) > 0
    assert np.std(data.participant_b) > 0


def test_npz_round_trip_loads_without_pickle(tmp_path) -> None:
    data = generate_synthetic_eeg(
        seed=4, condition=ExperimentalCondition.LOW_SHARED_COMPONENT
    )
    path = save_synthetic_eeg(data, tmp_path / "demo_synthetic_eeg.npz")
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["participant_a"], data.participant_a)
        np.testing.assert_array_equal(saved["participant_b"], data.participant_b)
        np.testing.assert_array_equal(saved["shared_component"], data.shared_component)
        np.testing.assert_array_equal(saved["time_seconds"], data.time_seconds)
        assert saved["sampling_frequency_hz"].item() == data.sampling_frequency_hz
        assert tuple(saved["channel_names"].tolist()) == data.channel_names
        assert saved["condition"].item() == "LOW_SHARED_COMPONENT"
        assert saved["random_seed"].item() == 4
        metadata = json.loads(saved["generation_parameters_json"].item())
        assert metadata["data_source"] == "synthetic"
        assert metadata["signal_unit"] == "V"
        assert metadata["random_seed"] == 4
        assert metadata["generation_parameters"] == data.generation_parameters


def test_npz_filename_must_preserve_synthetic_provenance(tmp_path) -> None:
    data = generate_synthetic_eeg(seed=5)
    with pytest.raises(ValueError, match="include 'synthetic'"):
        save_synthetic_eeg(data, tmp_path / "recording.npz")


@pytest.mark.parametrize("sampling_frequency_hz", [0, -256, 90, 90.7, np.nan, np.inf])
def test_invalid_sampling_frequency_is_rejected(sampling_frequency_hz: float) -> None:
    with pytest.raises(ValueError):
        generate_synthetic_eeg(sampling_frequency_hz=sampling_frequency_hz)


@pytest.mark.parametrize("duration_seconds", [0, -1, np.nan, np.inf, 0.001])
def test_invalid_duration_is_rejected(duration_seconds: float) -> None:
    with pytest.raises(ValueError):
        generate_synthetic_eeg(duration_seconds=duration_seconds)


@pytest.mark.parametrize("n_channels", [0, -1, 1.5, True])
def test_invalid_channel_count_is_rejected(n_channels: int) -> None:
    with pytest.raises(ValueError, match="n_channels"):
        generate_synthetic_eeg(n_channels=n_channels)


@pytest.mark.parametrize("condition", ["UNKNOWN", None])
def test_invalid_condition_is_rejected(condition: str | None) -> None:
    with pytest.raises(ValueError, match="condition"):
        generate_synthetic_eeg(condition=condition)
