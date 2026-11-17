"""Signal feature extraction for research experiments."""

from .spectral import (
    DEFAULT_BANDS_HZ,
    ChannelSpectralFeatures,
    SpectralConfig,
    SpectralFeatureResult,
    extract_spectral_features,
)

__all__ = [
    "DEFAULT_BANDS_HZ",
    "ChannelSpectralFeatures",
    "SpectralConfig",
    "SpectralFeatureResult",
    "extract_spectral_features",
]
