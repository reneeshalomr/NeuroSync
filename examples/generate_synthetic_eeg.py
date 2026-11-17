r"""Generate and inspect a synthetic two-participant EEG recording.

Run from the repository root with:
    .\.venv\Scripts\python.exe examples\generate_synthetic_eeg.py
Optionally pass an output path ending in .npz to save a compressed fixture.
"""

from __future__ import annotations

import argparse

from neurosync.data.synthetic_eeg import (
    ExperimentalCondition,
    generate_synthetic_eeg,
    save_synthetic_eeg,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="optional path for a synthetic NPZ file")
    args = parser.parse_args()

    data = generate_synthetic_eeg(
        sampling_frequency_hz=256,
        duration_seconds=5,
        n_channels=4,
        seed=2026,
        condition=ExperimentalCondition.HIGH_SHARED_COMPONENT,
    )
    print("Data source: synthetic (not a human EEG recording)")
    print(f"Condition: {data.condition.value}")
    print(f"Sampling frequency: {data.sampling_frequency_hz:g} Hz")
    print(f"Channels: {', '.join(data.channel_names)}")
    print(f"Participant A shape: {data.participant_a.shape}; unit: V")
    print(f"Participant B shape: {data.participant_b.shape}; unit: V")
    print(f"Time span: {data.time_seconds[0]:.3f} to {data.time_seconds[-1]:.3f} s")
    print(
        "Interpretation: injected shared components are not evidence of neural synchrony."
    )
    if args.output:
        saved_path = save_synthetic_eeg(data, args.output)
        print(f"Saved synthetic data to: {saved_path}")


if __name__ == "__main__":
    main()
