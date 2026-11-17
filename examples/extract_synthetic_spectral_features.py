r"""Extract descriptive spectral features from an unfiltered synthetic Raw.

Run from the repository root with:
    .\.venv\Scripts\python.exe examples\extract_synthetic_spectral_features.py

This illustrates Welch and band-power calculations only. It does not infer a
mental state, task label, or interpersonal synchrony.
"""

from __future__ import annotations

from neurosync.data.synthetic_eeg import generate_synthetic_eeg
from neurosync.features.spectral import extract_spectral_features
from neurosync.preprocessing import synthetic_participant_to_raw


def main() -> None:
    synthetic_pair = generate_synthetic_eeg(
        sampling_frequency_hz=256,
        duration_seconds=20,
        n_channels=4,
        seed=2026,
    )
    # No temporal filter is applied here so the demonstrated band estimates do
    # not inherit Day 2's illustrative 1–40 Hz preprocessing settings.
    raw = synthetic_participant_to_raw(synthetic_pair, participant="A")
    features = extract_spectral_features(raw)

    print("Data source: synthetic (not a human EEG recording)")
    print(f"Random seed: {synthetic_pair.random_seed}")
    print(f"Welch resolution: {features.metadata['frequency_resolution_hz']:g} Hz")
    print(f"PSD unit: {features.metadata['psd_unit']}")
    print(f"Absolute band power unit: {features.metadata['absolute_band_power_unit']}")
    print("Channel-level features:")
    for channel in features.channel_features:
        print(
            f"  {channel.channel_name}: relative alpha power="
            f"{channel.relative_band_power['alpha']:.4f}, "
            f"normalized spectral entropy="
            f"{channel.normalized_spectral_entropy:.4f}"
        )
    print("Across-channel mean and population-standard-deviation summaries:")
    for name, value in features.channel_summary.items():
        print(f"  {name}: {value:.8g}")
    print("Interpretation: descriptive signal features; no psychological inference.")


if __name__ == "__main__":
    main()
