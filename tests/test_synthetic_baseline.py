"""Tests for grouped classification of the synthetic generator condition."""

from __future__ import annotations

import json
import runpy

import numpy as np
import pytest

from neurosync.modeling.synthetic_baseline import (
    build_synthetic_condition_dataset,
    evaluate_synthetic_baselines,
    evaluate_synthetic_baselines_cv,
)


@pytest.fixture(scope="module")
def dataset():
    return build_synthetic_condition_dataset(
        n_sessions=8,
        seed=900,
        duration_seconds=6,
        n_channels=2,
    )


def test_dataset_contract_is_balanced_paired_and_finite(dataset) -> None:
    assert dataset.features.shape == (16, len(dataset.feature_names))
    assert dataset.labels.shape == dataset.groups.shape == (16,)
    assert np.isfinite(dataset.features).all()
    assert set(dataset.labels) == {"LOW_SHARED_COMPONENT", "HIGH_SHARED_COMPONENT"}
    assert dataset.metadata["data_source"] == "synthetic"
    assert dataset.metadata["not_human_collaboration_state"] is True
    for group in np.unique(dataset.groups):
        assert set(dataset.labels[dataset.groups == group]) == {
            "LOW_SHARED_COMPONENT",
            "HIGH_SHARED_COMPONENT",
        }


def test_dataset_is_reproducible_for_same_parameters(dataset) -> None:
    repeated = build_synthetic_condition_dataset(
        n_sessions=8,
        seed=900,
        duration_seconds=6,
        n_channels=2,
    )
    np.testing.assert_array_equal(dataset.features, repeated.features)
    np.testing.assert_array_equal(dataset.labels, repeated.labels)
    np.testing.assert_array_equal(dataset.groups, repeated.groups)
    assert dataset.metadata == repeated.metadata


def test_grouped_evaluation_has_balanced_classes_and_no_seed_leakage(dataset) -> None:
    result = evaluate_synthetic_baselines(dataset, random_state=5, n_splits=4)
    assert set(result.train_groups).isdisjoint(result.test_groups)
    assert len(result.test_groups) == 2
    assert result.class_balance == {
        "LOW_SHARED_COMPONENT": 8,
        "HIGH_SHARED_COMPONENT": 8,
    }
    assert result.metadata["scaler_fit_scope"].startswith("training fold only")
    assert result.metadata["not_human_collaboration_state"] is True
    assert set(result.metrics) == {
        "majority_dummy",
        "standardized_logistic_regression",
    }
    for model_metrics in result.metrics.values():
        assert 0 <= model_metrics["accuracy"] <= 1
        assert 0 <= model_metrics["balanced_accuracy"] <= 1
        assert 0 <= model_metrics["macro_f1"] <= 1
        assert np.asarray(model_metrics["confusion_matrix"]).shape == (2, 2)
        assert 0 <= model_metrics["paired_group_exact_accuracy"] <= 1
        low, high = model_metrics["paired_group_exact_accuracy_95pct_wilson_interval"]
        assert 0 <= low <= high <= 1
    assert (
        result.metrics["majority_dummy"][
            "paired_group_exact_accuracy_95pct_wilson_interval"
        ][0]
        == 0.0
    )


def test_grouped_evaluation_is_reproducible(dataset) -> None:
    first = evaluate_synthetic_baselines(dataset, random_state=11)
    second = evaluate_synthetic_baselines(dataset, random_state=11)
    assert first.train_groups == second.train_groups
    assert first.test_groups == second.test_groups
    assert first.metrics == second.metrics
    assert first.metadata == second.metadata


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_sessions": 3},
        {"n_channels": 0},
        {"duration_seconds": 1.5},
        {"sampling_frequency_hz": 90.0},
        {"seed": True},
    ],
)
def test_dataset_builder_rejects_invalid_parameters(kwargs) -> None:
    with pytest.raises((ValueError, TypeError)):
        build_synthetic_condition_dataset(**kwargs)


def test_evaluation_rejects_too_many_folds(dataset) -> None:
    with pytest.raises(ValueError, match="number of independent seed groups"):
        evaluate_synthetic_baselines(dataset, n_splits=9)


def test_cross_validation_assigns_each_pair_to_one_test_fold(dataset) -> None:
    result = evaluate_synthetic_baselines_cv(dataset, random_state=21, n_splits=4)
    flattened = [group for fold in result.fold_test_groups for group in fold]
    assert len(flattened) == len(set(flattened)) == 8
    assert set(flattened) == set(dataset.groups)
    assert result.class_balance == {
        "LOW_SHARED_COMPONENT": 8,
        "HIGH_SHARED_COMPONENT": 8,
    }
    assert (
        result.metadata["splitter"]
        == "StratifiedGroupKFold; all out-of-fold predictions"
    )
    for name, prediction in result.out_of_fold_predictions.items():
        assert prediction.shape == dataset.labels.shape
        assert set(prediction) <= set(dataset.labels)
        metrics = result.aggregate_metrics[name]
        assert 0 <= metrics["accuracy"] <= 1
        assert np.asarray(metrics["confusion_matrix"]).sum() == dataset.labels.size
        low, high = metrics["paired_group_exact_accuracy_95pct_wilson_interval"]
        assert 0 <= low <= high <= 1


def test_cross_validation_is_reproducible(dataset) -> None:
    first = evaluate_synthetic_baselines_cv(dataset, random_state=8)
    second = evaluate_synthetic_baselines_cv(dataset, random_state=8)
    assert first.fold_test_groups == second.fold_test_groups
    assert first.aggregate_metrics == second.aggregate_metrics
    for model in first.out_of_fold_predictions:
        np.testing.assert_array_equal(
            first.out_of_fold_predictions[model],
            second.out_of_fold_predictions[model],
        )


def test_end_to_end_workflow_writes_only_report_and_figures(tmp_path) -> None:
    workflow = runpy.run_path("examples/run_synthetic_research_workflow.py")
    report = workflow["run_workflow"](tmp_path, n_sessions=4)
    saved_report = json.loads((tmp_path / "synthetic_workflow_report.json").read_text())
    assert report == saved_report
    assert saved_report["signal_demo"]["data_source"] == "synthetic"
    assert (
        saved_report["synthetic_condition_evaluation"]["metadata"][
            "not_human_collaboration_state"
        ]
        is True
    )
    assert set(saved_report["generated_figures"]) == {
        "signals_synthetic.png",
        "spectral_synthetic.png",
        "coherence_synthetic.png",
    }
    for image_name in saved_report["generated_figures"]:
        assert (tmp_path / image_name).is_file()
    assert not list(tmp_path.glob("*.npz"))
