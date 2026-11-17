"""Scientific contract tests for Welch PSD and spectral features."""

import json

import mne
import numpy as np
import pytest
from scipy.signal import welch

from neurosync.data.synthetic_eeg import generate_synthetic_eeg
from neurosync.features.spectral import (
    SpectralConfig,
    extract_spectral_features,
)
from neurosync.preprocessing import (
    PreprocessingConfig,
    filter_eeg,
    synthetic_participant_to_raw,
)


def _synthetic_raw(
    signals: np.ndarray,
    sampling_frequency_hz: float = 256.0,
    channel_types: list[str] | None = None,
    data_source: str = "synthetic",
) -> mne.io.RawArray:
    info = mne.create_info(
        ch_names=[f"EEG{index + 1:03d}" for index in range(signals.shape[0])],
        sfreq=sampling_frequency_hz,
        ch_types=channel_types or ["eeg"] * signals.shape[0],
    )
    info["description"] = json.dumps({"data_source": data_source})
    return mne.io.RawArray(signals, info, verbose="ERROR")


@pytest.fixture
def two_tone_raw() -> mne.io.RawArray:
    sfreq = 256.0
    time_seconds = np.arange(4096) / sfreq
    signals = np.vstack(
        [
            4e-6 * np.sin(2 * np.pi * 10 * time_seconds),
            2e-6 * np.sin(2 * np.pi * 10 * time_seconds),
        ]
    )
    return _synthetic_raw(signals, sampling_frequency_hz=sfreq)


def test_welch_psd_matches_scipy_reference(two_tone_raw: mne.io.RawArray) -> None:
    config = SpectralConfig(window_seconds=2.0, overlap_fraction=0.5)
    result = extract_spectral_features(two_tone_raw, config)
    expected_frequency, expected_psd = welch(
        two_tone_raw.get_data(),
        fs=256.0,
        window="hann",
        nperseg=512,
        noverlap=256,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        average="mean",
        axis=-1,
    )
    np.testing.assert_array_equal(result.frequencies_hz, expected_frequency)
    np.testing.assert_allclose(result.psd_v2_per_hz, expected_psd)
    assert result.metadata["psd_method"] == "scipy.signal.welch"
    assert result.metadata["psd_unit"] == "V^2/Hz"
    assert result.metadata["frequency_resolution_hz"] == 0.5
    assert result.metadata["nfft_samples"] == 512
    assert result.metadata["source_participant_label"] is None


def test_band_power_relative_power_and_channel_summary_are_in_volts_squared(
    two_tone_raw: mne.io.RawArray,
) -> None:
    result = extract_spectral_features(two_tone_raw)
    first, second = result.channel_features

    assert first.channel_name == "EEG001"
    assert second.channel_name == "EEG002"
    assert first.absolute_band_power_v2["alpha"] == pytest.approx(
        (4e-6**2) / 2, rel=0.01
    )
    assert second.absolute_band_power_v2["alpha"] == pytest.approx(
        (2e-6**2) / 2, rel=0.01
    )
    assert first.relative_band_power["alpha"] > 0.99
    assert sum(first.relative_band_power.values()) == pytest.approx(1.0, abs=1e-8)
    assert result.channel_summary["channel_count"] == 2.0
    assert result.channel_summary["mean_alpha_power_v2"] == pytest.approx(
        5e-12, rel=0.01
    )
    assert result.channel_summary["std_alpha_power_v2"] == pytest.approx(
        3e-12, rel=0.02
    )


def test_spectral_entropy_is_finite_normalized_and_distinguishes_tone_from_noise(
    two_tone_raw: mne.io.RawArray,
) -> None:
    tone_result = extract_spectral_features(two_tone_raw)
    rng = np.random.Generator(np.random.PCG64(81))
    noise = _synthetic_raw(rng.normal(0.0, 4e-6, size=(1, 4096)))
    noise_result = extract_spectral_features(noise)

    tone_entropy = tone_result.channel_features[0].normalized_spectral_entropy
    noise_entropy = noise_result.channel_features[0].normalized_spectral_entropy
    assert 0.0 <= tone_entropy <= 1.0
    assert 0.0 <= noise_entropy <= 1.0
    assert tone_result.channel_features[0].spectral_entropy_nats > 0.0
    assert noise_entropy > tone_entropy
    in_range = (tone_result.frequencies_hz >= 0.5) & (
        tone_result.frequencies_hz <= 45.0
    )
    probabilities = tone_result.psd_v2_per_hz[0, in_range]
    probabilities = probabilities / probabilities.sum()
    expected_entropy = -np.sum(probabilities * np.log(probabilities))
    assert tone_result.channel_features[0].spectral_entropy_nats == pytest.approx(
        expected_entropy
    )
    assert tone_entropy == pytest.approx(expected_entropy / np.log(probabilities.size))
    assert noise_result.metadata["spectral_entropy_definition"] == (
        "Shannon entropy of normalized PSD-bin mass"
    )


def test_features_run_on_mne_preprocessed_synthetic_recording() -> None:
    data = generate_synthetic_eeg(
        sampling_frequency_hz=256,
        duration_seconds=10,
        n_channels=4,
        seed=32,
    )
    raw = synthetic_participant_to_raw(data)
    processed = filter_eeg(raw, PreprocessingConfig(average_reference=True))
    result = extract_spectral_features(processed)

    assert len(result.channel_features) == 4
    assert result.metadata["data_source"] == "synthetic"
    assert result.metadata["source_preprocessing"]["average_reference"] is True
    assert set(result.channel_summary) >= {
        "mean_spectral_entropy_nats",
        "mean_normalized_spectral_entropy",
        "mean_alpha_relative_power",
    }


def test_channels_marked_bad_are_excluded_from_channel_summaries(
    two_tone_raw: mne.io.RawArray,
) -> None:
    two_tone_raw.info["bads"] = ["EEG002"]
    result = extract_spectral_features(two_tone_raw)

    assert [item.channel_name for item in result.channel_features] == ["EEG001"]
    assert result.channel_summary["channel_count"] == 1.0
    assert result.metadata["excluded_bad_channels"] == ["EEG002"]


def test_spectral_features_are_deterministic(two_tone_raw: mne.io.RawArray) -> None:
    first = extract_spectral_features(two_tone_raw)
    second = extract_spectral_features(two_tone_raw)
    np.testing.assert_array_equal(first.psd_v2_per_hz, second.psd_v2_per_hz)
    assert first.channel_features == second.channel_features
    assert first.channel_summary == second.channel_summary
    assert first.metadata == second.metadata


def test_config_rejects_invalid_values_at_construction() -> None:
    with pytest.raises(ValueError):
        SpectralConfig(window_seconds=0)
    with pytest.raises(ValueError):
        SpectralConfig(overlap_fraction=1)
    with pytest.raises(ValueError):
        SpectralConfig(analysis_range_hz=(10.0, 10.0))
    with pytest.raises(ValueError):
        SpectralConfig(bands_hz=(("alpha", 8.0, 13.0), ("beta", 12.0, 30.0)))
    with pytest.raises(ValueError):
        SpectralConfig(bands_hz=(("outside", 0.1, 0.4),))
    with pytest.raises(ValueError):
        SpectralConfig(bands_hz=(("alpha", 8.0, 13.0), ("alpha", 13.0, 30.0)))


def test_analysis_range_must_fit_below_nyquist() -> None:
    low_rate_raw = _synthetic_raw(np.ones((1, 1024)) * 1e-6, sampling_frequency_hz=80.0)
    with pytest.raises(ValueError, match="Nyquist"):
        extract_spectral_features(low_rate_raw)


def test_only_synthetic_provenance_is_accepted(two_tone_raw: mne.io.RawArray) -> None:
    human_labeled = two_tone_raw.copy()
    human_labeled.info["description"] = json.dumps({"data_source": "human"})
    with pytest.raises(ValueError, match="labeled synthetic"):
        extract_spectral_features(human_labeled)


def test_malformed_provenance_shape_is_rejected(two_tone_raw: mne.io.RawArray) -> None:
    malformed = two_tone_raw.copy()
    malformed.info["description"] = "[]"
    with pytest.raises(TypeError, match="JSON object"):
        extract_spectral_features(malformed)


def test_zero_power_and_insufficient_resolution_are_rejected() -> None:
    zero_raw = _synthetic_raw(np.zeros((1, 4096)))
    with pytest.raises(ValueError, match="zero or invalid"):
        extract_spectral_features(zero_raw)

    short_time = np.arange(100) / 256.0
    short_raw = _synthetic_raw(3e-6 * np.sin(2 * np.pi * 3 * short_time)[None, :])
    with pytest.raises(ValueError, match="fewer than 2 bins in band"):
        extract_spectral_features(short_raw)


def test_recording_without_eeg_channels_is_rejected() -> None:
    eog_raw = _synthetic_raw(
        np.ones((1, 4096)) * 1e-6,
        channel_types=["eog"],
    )
    with pytest.raises(ValueError, match="at least one EEG channel"):
        extract_spectral_features(eog_raw)
