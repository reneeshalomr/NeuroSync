"""Signal feature extraction for research experiments."""

from .spectral import (
    DEFAULT_BANDS_HZ,
    ChannelSpectralFeatures,
    SpectralConfig,
    SpectralFeatureResult,
    extract_spectral_features,
)
from .unified import UnifiedFeatureTable, extract_unified_features

__all__ = [
    "DEFAULT_BANDS_HZ",
    "ChannelSpectralFeatures",
    "SpectralConfig",
    "SpectralFeatureResult",
    "UnifiedFeatureTable",
    "extract_spectral_features",
    "extract_unified_features",
]
