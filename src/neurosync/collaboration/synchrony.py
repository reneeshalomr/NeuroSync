"""Descriptive cross-signal magnitude-squared coherence for synthetic EEG.

Coherence is a frequency-specific statistical association. This function does
not test interpersonal synchrony, causal coupling, or a collaboration state.
No surrogate inference or significance correction is performed.
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
from scipy.signal import coherence


@dataclass(frozen=True)
class SynchronyConfig:
    """Welch coherence settings; window length controls frequency resolution."""

    window_seconds: float = 2.0
    overlap_fraction: float = 0.5
    bands_hz: tuple[tuple[str, float, float], ...] = (
        ("delta", 0.5, 4.0),
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 13.0),
        ("beta", 13.0, 30.0),
        ("gamma", 30.0, 45.0),
    )

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
            or not 0 <= self.overlap_fraction < 1
        ):
            raise ValueError("overlap_fraction must be in [0, 1)")
        names: set[str] = set()
        last_high = 0.0
        for name, low, high in self.bands_hz:
            if not name or name in names:
                raise ValueError("band names must be unique and non-empty")
            if (
                not math.isfinite(low)
                or not math.isfinite(high)
                or low < 0
                or low >= high
            ):
                raise ValueError(
                    "bands must have finite, non-negative increasing limits"
                )
            if low < last_high:
                raise ValueError("bands must be ordered and non-overlapping")
            names.add(name)
            last_high = high


@dataclass(frozen=True)
class SynchronyResult:
    """Matched-channel coherence and its descriptive frequency-band summaries."""

    frequencies_hz: NDArray[np.float64]
    channel_names: tuple[str, ...]
    coherence: NDArray[np.float64]
    band_mean_coherence: dict[str, float]
    channel_band_mean_coherence: dict[str, dict[str, float]]
    metadata: dict[str, Any]


def _synthetic_description(raw: mne.io.BaseRaw, label: str) -> dict[str, Any]:
    description = raw.info.get("description")
    try:
        metadata = json.loads(description) if isinstance(description, str) else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"participant {label} has invalid provenance JSON") from exc
    if metadata.get("data_source") != "synthetic":
        raise ValueError(f"participant {label} must be explicitly labeled synthetic")
    if metadata.get("signal_unit") != "V":
        raise ValueError(f"participant {label} must declare signal_unit as 'V'")
    return metadata


def compute_synthetic_coherence(
    raw_a: mne.io.BaseRaw,
    raw_b: mne.io.BaseRaw,
    config: SynchronyConfig = SynchronyConfig(),
) -> SynchronyResult:
    """Compute Welch magnitude-squared coherence for matched synthetic EEG.

    Only EEG channels with matching names and not marked bad in either recording
    are analyzed. Inputs must have equal sample counts and sampling rates. The
    result is descriptive; finite-sample coherence is biased upward and no
    surrogate/control inference or correction for multiple comparisons occurs.
    """
    provenance_a = _synthetic_description(raw_a, "A")
    provenance_b = _synthetic_description(raw_b, "B")
    sfreq_a, sfreq_b = float(raw_a.info["sfreq"]), float(raw_b.info["sfreq"])
    if not math.isclose(sfreq_a, sfreq_b, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("participants must have identical sampling frequencies")
    if raw_a.n_times != raw_b.n_times:
        raise ValueError("participants must have identical sample counts")
    if not np.isfinite(sfreq_a) or sfreq_a <= 0:
        raise ValueError("sampling frequency must be finite and positive")
    names_a = set(raw_a.copy().pick("eeg").ch_names) - set(raw_a.info["bads"])
    names_b = set(raw_b.copy().pick("eeg").ch_names) - set(raw_b.info["bads"])
    channels = tuple(name for name in raw_a.ch_names if name in names_a & names_b)
    if not channels:
        raise ValueError("no matched, non-bad EEG channels are available")
    nperseg = round(config.window_seconds * sfreq_a)
    if nperseg < 2 or nperseg > raw_a.n_times:
        raise ValueError(
            "window_seconds must produce at least 2 samples and fit the data"
        )
    nyquist = sfreq_a / 2
    if any(high > nyquist for _, _, high in config.bands_hz):
        raise ValueError("all band limits must be at or below Nyquist")
    noverlap = math.floor(nperseg * config.overlap_fraction)
    data_a = raw_a.get_data(picks=list(channels), reject_by_annotation=None)
    data_b = raw_b.get_data(picks=list(channels), reject_by_annotation=None)
    if not np.isfinite(data_a).all() or not np.isfinite(data_b).all():
        raise ValueError("EEG samples must be finite")
    per_channel: list[NDArray[np.float64]] = []
    frequencies: NDArray[np.float64] | None = None
    for signal_a, signal_b in zip(data_a, data_b, strict=True):
        freq, values = coherence(
            signal_a,
            signal_b,
            fs=sfreq_a,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nperseg,
            detrend="constant",
            axis=-1,
        )
        frequencies = freq
        per_channel.append(values)
    assert frequencies is not None
    matrix = np.stack(per_channel)
    if not np.isfinite(matrix).all():
        raise ValueError("coherence is undefined for one or more zero-power channels")
    channel_bands: dict[str, dict[str, float]] = {}
    for index, name in enumerate(channels):
        channel_bands[name] = {}
        for band_name, low, high in config.bands_hz:
            mask = (frequencies >= low) & (frequencies <= high)
            if not mask.any():
                raise ValueError(
                    f"no coherence frequency bins fall in band {band_name!r}"
                )
            channel_bands[name][band_name] = float(matrix[index, mask].mean())
    band_means = {
        band: float(np.mean([values[band] for values in channel_bands.values()]))
        for band, _, _ in config.bands_hz
    }
    return SynchronyResult(
        frequencies_hz=frequencies,
        channel_names=channels,
        coherence=matrix,
        band_mean_coherence=band_means,
        channel_band_mean_coherence=channel_bands,
        metadata={
            "statistic": "magnitude_squared_coherence",
            "interpretation": "descriptive frequency-specific statistical association",
            "data_source": "synthetic",
            "condition_a": provenance_a.get("condition"),
            "condition_b": provenance_b.get("condition"),
            "random_seed_a": provenance_a.get("random_seed"),
            "random_seed_b": provenance_b.get("random_seed"),
            "signal_unit": "V",
            "window": "hann",
            "window_seconds": nperseg / sfreq_a,
            "nperseg": nperseg,
            "noverlap": noverlap,
            "nfft": nperseg,
            "detrend": "constant",
            "frequency_resolution_hz": sfreq_a / nperseg,
            "bands_hz": {name: [low, high] for name, low, high in config.bands_hz},
            "excluded_bad_channels": {
                "A": [name for name in raw_a.info["bads"] if name in raw_a.ch_names],
                "B": [name for name in raw_b.info["bads"] if name in raw_b.ch_names],
            },
            "surrogate_or_control_inference": "none performed",
            "multiple_comparison_correction": "none",
            "finite_sample_bias": "coherence estimates may be biased upward",
            "scipy_version": scipy.__version__,
        },
    )
