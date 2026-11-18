"""Baseline models for explicitly synthetic NeuroSync research fixtures."""

from neurosync.modeling.multimodal import (
    MultimodalComparisonResult,
    MultimodalCVConfig,
    evaluate_multimodal_comparison,
)
from neurosync.modeling.synthetic_baseline import (
    SyntheticBaselineEvaluation,
    SyntheticConditionDataset,
    SyntheticCrossValidationEvaluation,
    build_synthetic_condition_dataset,
    evaluate_synthetic_baselines,
    evaluate_synthetic_baselines_cv,
)

__all__ = [
    "MultimodalCVConfig",
    "MultimodalComparisonResult",
    "SyntheticBaselineEvaluation",
    "SyntheticConditionDataset",
    "SyntheticCrossValidationEvaluation",
    "build_synthetic_condition_dataset",
    "evaluate_multimodal_comparison",
    "evaluate_synthetic_baselines",
    "evaluate_synthetic_baselines_cv",
]
