"""Baseline models for explicitly synthetic NeuroSync research fixtures."""

from neurosync.modeling.synthetic_baseline import (
    SyntheticBaselineEvaluation,
    SyntheticConditionDataset,
    SyntheticCrossValidationEvaluation,
    build_synthetic_condition_dataset,
    evaluate_synthetic_baselines,
    evaluate_synthetic_baselines_cv,
)

__all__ = [
    "SyntheticBaselineEvaluation",
    "SyntheticConditionDataset",
    "SyntheticCrossValidationEvaluation",
    "build_synthetic_condition_dataset",
    "evaluate_synthetic_baselines",
    "evaluate_synthetic_baselines_cv",
]
