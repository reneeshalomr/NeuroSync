"""Explicit offline EEG filtering and referencing with MNE."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import mne
import scipy


@dataclass(frozen=True)
class PreprocessingConfig:
    """Filter and reference choices for one offline preprocessing pass.

    The defaults apply a 1–40 Hz zero-phase FIR band-pass and average
    reference. These are explicit engineering defaults, not universally
    optimal EEG settings. Notch filtering is opt-in.
    """

    highpass_hz: float | None = 1.0
    lowpass_hz: float | None = 40.0
    notch_frequencies_hz: tuple[float, ...] = ()
    average_reference: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("highpass_hz", self.highpass_hz),
            ("lowpass_hz", self.lowpass_hz),
        ):
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number or None")
        if self.highpass_hz is None and self.lowpass_hz is None:
            raise ValueError("at least one of highpass_hz or lowpass_hz must be set")
        if (
            self.highpass_hz is not None
            and self.lowpass_hz is not None
            and self.highpass_hz >= self.lowpass_hz
        ):
            raise ValueError("highpass_hz must be lower than lowpass_hz")
        for frequency in self.notch_frequencies_hz:
            if (
                isinstance(frequency, bool)
                or not math.isfinite(frequency)
                or frequency <= 0
            ):
                raise ValueError("notch frequencies must be finite positive numbers")
        if not isinstance(self.average_reference, bool):
            raise TypeError("average_reference must be a boolean")


def filter_eeg(
    raw: mne.io.BaseRaw,
    config: PreprocessingConfig | None = None,
) -> mne.io.BaseRaw:
    """Return a filtered copy of an MNE recording; the input is unchanged.

    Filtering is offline and zero-phase. It can smear transients around event
    times and should not be interpreted as a real-time pipeline. The caller
    chooses notch frequencies explicitly; no mains frequency is assumed.
    """
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("raw must be an MNE Raw instance")
    selected = config or PreprocessingConfig()
    sfreq = float(raw.info["sfreq"])
    nyquist_hz = sfreq / 2.0
    for name, value in (
        ("highpass_hz", selected.highpass_hz),
        ("lowpass_hz", selected.lowpass_hz),
    ):
        if value is not None and value >= nyquist_hz:
            raise ValueError(f"{name} must be below Nyquist ({nyquist_hz:g} Hz)")
    if any(frequency >= nyquist_hz for frequency in selected.notch_frequencies_hz):
        raise ValueError(f"notch frequencies must be below Nyquist ({nyquist_hz:g} Hz)")

    processed = raw.copy()
    processed.filter(
        l_freq=selected.highpass_hz,
        h_freq=selected.lowpass_hz,
        picks="eeg",
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose="ERROR",
    )
    if selected.notch_frequencies_hz:
        processed.notch_filter(
            freqs=list(selected.notch_frequencies_hz),
            picks="eeg",
            method="fir",
            phase="zero",
            fir_design="firwin",
            verbose="ERROR",
        )
    if selected.average_reference:
        processed.set_eeg_reference(
            ref_channels="average", projection=False, verbose="ERROR"
        )

    try:
        provenance = json.loads(processed.info.get("description") or "{}")
    except json.JSONDecodeError:
        provenance = {"prior_description": processed.info.get("description")}
    provenance["preprocessing"] = {
        "highpass_hz": selected.highpass_hz,
        "lowpass_hz": selected.lowpass_hz,
        "notch_frequencies_hz": list(selected.notch_frequencies_hz),
        "average_reference": selected.average_reference,
        "method": "MNE zero-phase FIR, firwin; offline",
        "mne_version": mne.__version__,
        "scipy_version": scipy.__version__,
    }
    processed.info["description"] = json.dumps(provenance, sort_keys=True)
    return processed
