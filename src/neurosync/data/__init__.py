"""Data contracts and explicitly synthetic data generators."""

from .synthetic_eeg import (
    ExperimentalCondition,
    SyntheticEEGData,
    generate_synthetic_eeg,
    save_synthetic_eeg,
)

__all__ = [
    "ExperimentalCondition",
    "SyntheticEEGData",
    "generate_synthetic_eeg",
    "save_synthetic_eeg",
]
