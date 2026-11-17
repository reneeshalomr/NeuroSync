"""Data contracts and explicitly synthetic data generators."""

from .event_driven import generate_event_modulated_synthetic_eeg
from .synthetic_eeg import (
    ExperimentalCondition,
    SyntheticEEGData,
    generate_synthetic_eeg,
    save_synthetic_eeg,
)

__all__ = [
    "ExperimentalCondition",
    "SyntheticEEGData",
    "generate_event_modulated_synthetic_eeg",
    "generate_synthetic_eeg",
    "save_synthetic_eeg",
]
