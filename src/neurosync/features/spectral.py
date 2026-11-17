"""Welch PSD, band power, and spectral entropy for synthetic MNE recordings.

Absolute band power integrates a one-sided Welch power spectral density and is
reported in V². Relative band power divides by total power integrated across
the configured analysis range. Spectral entropy is Shannon entropy over
normalized PSD-bin mass; both nats and a [0, 1] normalized value are returned.

These are descriptive signal statistics. They do not directly reveal mental
states, emotion, intention, collaboration quality, or interpersonal synchrony.
This first API accepts only MNE objects marked as synthetic in their metadata.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import mne
import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.integrate import trapezoid
from scipy.signal import welch

DEFAULT_BANDS_HZ: tuple[tuple[str, float, float], ...] = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
)


@dataclass(frozen=True)
class SpectralConfig:
    """Welch and band definitions for a spectral feature pass.

    Welch uses a Hann window, constant detrending, mean averaging, one-sided
    density scaling, and 50% overlap by default. These choices are explicit in
    result metadata. ``analysis_range_hz`` is also the denominator range for
    relative band power and the support for spectral entropy.
    """

    window_seconds: float = 2.0
    overlap_fraction: float = 0.5
    analysis_range_hz: tuple[float, float] = (0.5, 45.0)
    bands_hz: tuple[tuple[str, float, float], ...] = DEFAULT_BANDS_HZ

    def __post_init__(self) -> None:
        if (
            isinstance(self.window_seconds, bool)
            or not math.isfinite(self.window_seconds)
            or self.window_seconds <= 0
        ):
            raise ValueError("window_seconds must be finite and positive")
        if (
            isinstance(self.overlap_fraction, bool)
            or not math.isfinite(self.overlap_fraction)
            or not 0.0 <= self.overlap_fraction < 1.0
        ):
            raise ValueError("overlap_fraction must be in [0, 1)")
        if len(self.analysis_range_hz) != 2:
            raise ValueError("analysis_range_hz must contain a low and high frequency")
        low, high = self.analysis_range_hz
        if not math.isfinite(low) or not math.isfinite(high) or low < 0 or low >= high:
            raise ValueError(
                "analysis_range_hz must be finite, non-negative, and increasing"
            )
        if not self.bands_hz:
            raise ValueError("at least one frequency band is required")
        names: set[str] = set()
        previous_high = low
        for name, band_low, band_high in self.bands_hz:
            if not isinstance(name, str) or not name:
                raise ValueError("frequency band names must be non-empty strings")
            if name in names:
                raise ValueError(f"frequency band name {name!r} is duplicated")
            names.add(name)
            if (
                not math.isfinite(band_low)
                or not math.isfinite(band_high)
                or band_low < low
                or band_low >= band_high
                or band_high > high
            ):
                raise ValueError(
                    f"band {name!r} must be increasing and within analysis_range_hz"
                )
            if band_low < previous_high:
                raise ValueError("frequency bands must be ordered and non-overlapping")
            previous_high = band_high


@dataclass(frozen=True)
class ChannelSpectralFeatures:
    """Band powers and entropy estimates for one EEG channel."""

    channel_name: str
    absolute_band_power_v2: dict[str, float]
    relative_band_power: dict[str, float]
    spectral_entropy_nats: float
    normalized_spectral_entropy: float


@dataclass(frozen=True)
class SpectralFeatureResult:
    """Welch spectrum, per-channel features, summaries, and provenance."""

    channel_features: tuple[ChannelSpectralFeatures, ...]
    channel_summary: dict[str, float]
    frequencies_hz: NDArray[np.float64]
    psd_v2_per_hz: NDArray[np.float64]
    metadata: dict[str, Any]


def _entropy_from_psd(psd: NDArray[np.float64]) -> tuple[float, float]:
    total = float(np.sum(psd))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("spectral entropy is undefined for zero or invalid PSD power")
    probabilities = psd / total
    positive = probabilities > 0
    entropy_nats = float(
        -np.sum(probabilities[positive] * np.log(probabilities[positive]))
    )
    normalized = entropy_nats / math.log(probabilities.size)
    return entropy_nats, float(normalized)


def extract_spectral_features(
    raw: mne.io.BaseRaw,
    config: SpectralConfig | None = None,
) -> SpectralFeatureResult:
    """Compute Welch PSD and channel-level spectral features from synthetic EEG.

    Args:
        raw: MNE Raw object whose ``info['description']`` JSON has
            ``data_source='synthetic'``. Data are read in volts.
        config: Windowing, analysis-range, and band settings. Defaults are a
            two-second Hann window, 50% overlap, and standard EEG band limits.

    Returns:
        Full one-sided Welch PSD in V²/Hz, per-channel absolute and relative
        band powers, spectral entropy in nats and normalized to [0, 1],
        channel-wise mean/population-standard-deviation summaries, and
        reproducibility metadata.

    Raises:
        TypeError: If ``raw`` is not an MNE Raw instance.
        ValueError: For missing synthetic provenance, invalid signal data,
            invalid Nyquist support, insufficient frequency bins, or zero PSD.
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("raw must be an MNE Raw instance")
    selected = config or SpectralConfig()
    try:
        provenance = json.loads(raw.info.get("description") or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(
            "MNE Raw description must contain synthetic provenance JSON"
        ) from exc
    if not isinstance(provenance, dict):
        raise TypeError("MNE Raw description must contain a JSON object")
    if provenance.get("data_source") != "synthetic":
        raise ValueError("spectral features currently require data labeled synthetic")

    sfreq = float(raw.info["sfreq"])
    analysis_low, analysis_high = selected.analysis_range_hz
    if analysis_high > sfreq / 2.0:
        raise ValueError(
            f"analysis_range_hz upper bound must not exceed Nyquist ({sfreq / 2.0:g} Hz)"
        )
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if eeg_picks.size == 0:
        raise ValueError("recording must contain at least one EEG channel")
    signals = raw.get_data(picks=eeg_picks)
    if signals.ndim != 2 or signals.shape[1] < 2:
        raise ValueError(
            "EEG data must have shape (channels, samples) with at least 2 samples"
        )
    if not np.isfinite(signals).all():
        raise ValueError("EEG data must contain only finite values")

    nperseg = min(round(selected.window_seconds * sfreq), signals.shape[1])
    if nperseg < 2:
        raise ValueError(
            "window_seconds and sampling frequency must produce at least 2 samples"
        )
    noverlap = int(nperseg * selected.overlap_fraction)
    frequencies_hz, psd = welch(
        signals,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        average="mean",
        axis=-1,
    )
    range_mask = (frequencies_hz >= analysis_low) & (frequencies_hz <= analysis_high)
    if np.count_nonzero(range_mask) < 2:
        raise ValueError(
            "Welch resolution provides fewer than 2 bins in analysis_range_hz"
        )
    total_power = trapezoid(psd[:, range_mask], frequencies_hz[range_mask], axis=-1)
    if not np.isfinite(total_power).all() or np.any(total_power <= 0):
        raise ValueError(
            "relative power is undefined for zero or invalid analysis-range power"
        )

    per_channel: list[ChannelSpectralFeatures] = []
    for channel_index, channel_name in enumerate(raw.ch_names[i] for i in eeg_picks):
        absolute: dict[str, float] = {}
        relative: dict[str, float] = {}
        for band_name, band_low, band_high in selected.bands_hz:
            band_mask = (frequencies_hz >= band_low) & (frequencies_hz <= band_high)
            if np.count_nonzero(band_mask) < 2:
                raise ValueError(
                    f"Welch resolution provides fewer than 2 bins in band {band_name!r}"
                )
            power_v2 = float(
                trapezoid(psd[channel_index, band_mask], frequencies_hz[band_mask])
            )
            absolute[band_name] = power_v2
            relative[band_name] = power_v2 / float(total_power[channel_index])
        entropy_masked = psd[channel_index, range_mask]
        entropy_nats, entropy_normalized = _entropy_from_psd(entropy_masked)
        per_channel.append(
            ChannelSpectralFeatures(
                channel_name=channel_name,
                absolute_band_power_v2=absolute,
                relative_band_power=relative,
                spectral_entropy_nats=entropy_nats,
                normalized_spectral_entropy=entropy_normalized,
            )
        )

    summary: dict[str, float] = {"channel_count": float(len(per_channel))}
    for band_name, _, _ in selected.bands_hz:
        absolute_values = np.asarray(
            [item.absolute_band_power_v2[band_name] for item in per_channel]
        )
        relative_values = np.asarray(
            [item.relative_band_power[band_name] for item in per_channel]
        )
        summary[f"mean_{band_name}_power_v2"] = float(np.mean(absolute_values))
        summary[f"std_{band_name}_power_v2"] = float(np.std(absolute_values, ddof=0))
        summary[f"mean_{band_name}_relative_power"] = float(np.mean(relative_values))
        summary[f"std_{band_name}_relative_power"] = float(
            np.std(relative_values, ddof=0)
        )
    entropy_values = np.asarray([item.spectral_entropy_nats for item in per_channel])
    normalized_entropy_values = np.asarray(
        [item.normalized_spectral_entropy for item in per_channel]
    )
    summary["mean_spectral_entropy_nats"] = float(np.mean(entropy_values))
    summary["std_spectral_entropy_nats"] = float(np.std(entropy_values, ddof=0))
    summary["mean_normalized_spectral_entropy"] = float(
        np.mean(normalized_entropy_values)
    )
    summary["std_normalized_spectral_entropy"] = float(
        np.std(normalized_entropy_values, ddof=0)
    )

    metadata = {
        "data_source": "synthetic",
        "sampling_frequency_hz": sfreq,
        "signal_unit": "V",
        "psd_unit": "V^2/Hz",
        "absolute_band_power_unit": "V^2",
        "relative_power_definition": "band integral / total integral over analysis_range_hz",
        "spectral_entropy_definition": "Shannon entropy of normalized PSD-bin mass",
        "spectral_entropy_unit": "nats",
        "normalized_spectral_entropy_range": [0.0, 1.0],
        "psd_method": "scipy.signal.welch",
        "window": "hann",
        "detrend": "constant",
        "average": "mean",
        "scaling": "density",
        "nperseg_samples": nperseg,
        "nfft_samples": nperseg,
        "noverlap_samples": noverlap,
        "input_sample_count": int(signals.shape[1]),
        "frequency_resolution_hz": sfreq / nperseg,
        "analysis_range_hz": list(selected.analysis_range_hz),
        "bands_hz": {name: [low, high] for name, low, high in selected.bands_hz},
        "channel_summary_std_ddof": 0,
        "excluded_bad_channels": list(raw.info["bads"]),
        "source_condition": provenance.get("condition"),
        "source_random_seed": provenance.get("random_seed"),
        "source_participant_label": provenance.get("participant_label"),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "mne_version": mne.__version__,
        "source_preprocessing": provenance.get("preprocessing"),
    }
    return SpectralFeatureResult(
        channel_features=tuple(per_channel),
        channel_summary=summary,
        frequencies_hz=frequencies_hz,
        psd_v2_per_hz=psd,
        metadata=metadata,
    )
