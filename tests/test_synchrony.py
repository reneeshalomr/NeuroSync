"""Contract tests for descriptive coherence on constructed signals."""

from __future__ import annotations

import json

import mne
import numpy as np
import pytest
from scipy.signal import coherence

from neurosync.collaboration import (
    SynchronyConfig,
    compute_synthetic_coherence,
)


def _raw(
    data: np.ndarray, sfreq: float = 128.0, *, source: str = "synthetic"
) -> mne.io.RawArray:
    info = mne.create_info([f"EEG{i}" for i in range(data.shape[0])], sfreq, "eeg")
    info["description"] = json.dumps(
        {
            "data_source": source,
            "condition": "fixture",
            "random_seed": 4,
            "signal_unit": "V",
        }
    )
    return mne.io.RawArray(data, info, verbose="ERROR")


def test_coherence_matches_scipy_reference_and_summaries() -> None:
    sfreq = 128.0
    time = np.arange(1280) / sfreq
    rng = np.random.default_rng(19)
    shared = np.sin(2 * np.pi * 10 * time)
    a = (
        np.stack(
            [shared + 0.2 * rng.normal(size=time.size), rng.normal(size=time.size)]
        )
        * 1e-6
    )
    b = (
        np.stack(
            [shared + 0.2 * rng.normal(size=time.size), rng.normal(size=time.size)]
        )
        * 1e-6
    )
    result = compute_synthetic_coherence(_raw(a), _raw(b))
    freq, expected = coherence(
        a[0],
        b[0],
        fs=sfreq,
        window="hann",
        nperseg=256,
        noverlap=128,
        nfft=256,
        detrend="constant",
    )
    np.testing.assert_array_equal(result.frequencies_hz, freq)
    np.testing.assert_allclose(result.coherence[0], expected)
    assert result.coherence.shape == (2, 129)
    assert result.channel_names == ("EEG0", "EEG1")
    assert set(result.band_mean_coherence) == {
        "delta",
        "theta",
        "alpha",
        "beta",
        "gamma",
    }
    alpha_mask = (freq >= 8.0) & (freq <= 13.0)
    assert result.channel_band_mean_coherence["EEG0"]["alpha"] == pytest.approx(
        expected[alpha_mask].mean()
    )
    assert result.metadata["surrogate_or_control_inference"] == "none performed"
    assert result.metadata["signal_unit"] == "V"


def test_exact_common_waveform_has_unit_coherence_at_its_frequency() -> None:
    sfreq = 128.0
    time = np.arange(2048) / sfreq
    wave = np.sin(2 * np.pi * 12 * time)[None, :] * 1e-6
    result = compute_synthetic_coherence(_raw(wave), _raw(wave.copy()))
    peak = np.argmin(np.abs(result.frequencies_hz - 12.0))
    assert result.coherence[0, peak] == pytest.approx(1.0)


def test_excludes_channels_marked_bad_in_either_recording() -> None:
    rng = np.random.default_rng(8)
    data = rng.normal(size=(2, 1024)) * 1e-6
    first, second = _raw(data), _raw(data.copy())
    first.info["bads"] = ["EEG0"]
    result = compute_synthetic_coherence(first, second)
    assert result.channel_names == ("EEG1",)
    assert result.metadata["excluded_bad_channels"]["A"] == ["EEG0"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_seconds": 0},
        {"overlap_fraction": 1},
        {"bands_hz": (("bad", 10, 5),)},
    ],
)
def test_rejects_invalid_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        config = SynchronyConfig(**kwargs)
        compute_synthetic_coherence(
            _raw(np.ones((1, 1024)) * 1e-6),
            _raw(np.ones((1, 1024)) * 1e-6),
            config,
        )


def test_rejects_non_synthetic_provenance() -> None:
    data = np.ones((1, 1024)) * 1e-6
    with pytest.raises(ValueError, match="synthetic"):
        compute_synthetic_coherence(_raw(data, source="human"), _raw(data))


def test_rejects_missing_volt_provenance() -> None:
    data = np.random.default_rng(3).normal(size=(1, 1024)) * 1e-6
    raw = _raw(data)
    raw.info["description"] = json.dumps({"data_source": "synthetic"})
    with pytest.raises(ValueError, match="signal_unit"):
        compute_synthetic_coherence(raw, _raw(data.copy()))


def test_rejects_mismatched_sampling_rate_or_length() -> None:
    data = np.ones((1, 1024)) * 1e-6
    with pytest.raises(ValueError, match="sampling frequencies"):
        compute_synthetic_coherence(_raw(data, 128), _raw(data, 100))
    with pytest.raises(ValueError, match="sample counts"):
        compute_synthetic_coherence(_raw(data), _raw(data[:, :-1]))


def test_rejects_window_above_recording_length_and_band_above_nyquist() -> None:
    data = np.ones((1, 1024)) * 1e-6
    with pytest.raises(ValueError, match="fit the data"):
        compute_synthetic_coherence(
            _raw(data), _raw(data), SynchronyConfig(window_seconds=20)
        )
    with pytest.raises(ValueError, match="Nyquist"):
        compute_synthetic_coherence(
            _raw(data), _raw(data), SynchronyConfig(bands_hz=(("high", 30, 70),))
        )


def test_rejects_zero_power_channels() -> None:
    data = np.zeros((1, 1024))
    with (
        pytest.warns(RuntimeWarning, match="invalid value"),
        pytest.raises(ValueError, match="zero-power"),
    ):
        compute_synthetic_coherence(_raw(data), _raw(data.copy()))


def test_results_repeat_exactly_for_same_inputs() -> None:
    rng = np.random.default_rng(33)
    a = _raw(rng.normal(size=(2, 2048)) * 1e-6)
    b = _raw(rng.normal(size=(2, 2048)) * 1e-6)
    first, second = compute_synthetic_coherence(a, b), compute_synthetic_coherence(a, b)
    np.testing.assert_array_equal(first.coherence, second.coherence)
    assert first.metadata == second.metadata
