"""MNE-backed preprocessing for explicitly synthetic EEG inputs."""

from .epochs import epoch_eeg
from .filtering import PreprocessingConfig, filter_eeg
from .raw import synthetic_participant_to_raw

__all__ = [
    "PreprocessingConfig",
    "epoch_eeg",
    "filter_eeg",
    "synthetic_participant_to_raw",
]
