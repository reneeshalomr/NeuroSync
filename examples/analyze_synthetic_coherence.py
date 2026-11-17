"""Compute descriptive coherence for a seeded synthetic participant pair."""

from __future__ import annotations

from neurosync.collaboration import compute_synthetic_coherence
from neurosync.data.synthetic_eeg import (
    ExperimentalCondition,
    generate_synthetic_eeg,
)
from neurosync.preprocessing.raw import synthetic_participant_to_raw


def main() -> None:
    data = generate_synthetic_eeg(
        sampling_frequency_hz=256,
        duration_seconds=20,
        n_channels=4,
        seed=2026,
        condition=ExperimentalCondition.HIGH_SHARED_COMPONENT,
    )
    raw_a = synthetic_participant_to_raw(data, "A")
    raw_b = synthetic_participant_to_raw(data, "B")
    result = compute_synthetic_coherence(raw_a, raw_b)
    print("Synthetic EEG only; descriptive magnitude-squared coherence")
    print(f"Channels: {', '.join(result.channel_names)}")
    print("Mean coherence by band:")
    for band, value in result.band_mean_coherence.items():
        print(f"  {band}: {value:.3f}")
    print(
        "These constructed data and statistics are not human evidence or a test of interpersonal synchrony."
    )


if __name__ == "__main__":
    main()
