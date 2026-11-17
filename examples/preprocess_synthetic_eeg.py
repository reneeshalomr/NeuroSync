r"""Run the Day 2 MNE preprocessing path on synthetic EEG only.

Run from the repository root with:
    .\.venv\Scripts\python.exe examples\preprocess_synthetic_eeg.py

The demonstration events are illustrative sample indices created for this
example; they are not behavioral annotations or experimental observations.
"""

from __future__ import annotations

import json

import mne
import numpy as np

from neurosync.data.synthetic_eeg import generate_synthetic_eeg
from neurosync.preprocessing import (
    PreprocessingConfig,
    epoch_eeg,
    filter_eeg,
    synthetic_participant_to_raw,
)


def main() -> None:
    data = generate_synthetic_eeg(
        sampling_frequency_hz=256,
        duration_seconds=8,
        n_channels=4,
        seed=2026,
    )
    raw = synthetic_participant_to_raw(data, participant="A")
    processed = filter_eeg(
        raw,
        PreprocessingConfig(
            highpass_hz=1.0,
            lowpass_hz=40.0,
            average_reference=True,
        ),
    )

    illustrative_event_times = (2.0, 5.0)
    events = np.array(
        [
            [round(time_s * data.sampling_frequency_hz), 0, 1]
            for time_s in illustrative_event_times
        ],
        dtype=np.int64,
    )
    epochs = epoch_eeg(
        processed,
        events,
        {"illustrative_synthetic_event": 1},
        tmin=-0.25,
        tmax=0.5,
    )

    provenance = json.loads(processed.info["description"])
    print("Source: synthetic signal fixture (not a human EEG recording)")
    print(f"MNE version: {mne.__version__}")
    print(f"Sampling frequency: {processed.info['sfreq']:g} Hz")
    print(f"Filtered channels: {', '.join(processed.ch_names)}")
    print(f"Filter settings: {provenance['preprocessing']}")
    print(
        f"Illustrative event count: {len(epochs)}; epoch array shape: {epochs.get_data().shape}"
    )
    print("Baseline correction: none")
    print(
        "Interpretation: software workflow demonstration only; no neural or task inference."
    )


if __name__ == "__main__":
    main()
