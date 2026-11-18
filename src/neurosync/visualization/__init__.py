"""Reusable, clearly labeled exploratory visualizations."""

from neurosync.visualization.collaboration import (
    plot_coherence,
    plot_spectral_features,
)
from neurosync.visualization.model_comparison import plot_model_comparison
from neurosync.visualization.signals import plot_synthetic_signals

__all__ = [
    "plot_coherence",
    "plot_model_comparison",
    "plot_spectral_features",
    "plot_synthetic_signals",
]
