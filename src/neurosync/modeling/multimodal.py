"""Group-aware comparative models for synthetic event-level feature tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neurosync.features.unified import UnifiedFeatureTable

EEG_FEATURE_PREFIXES = ("eeg_a_", "eeg_b_", "eeg_mean_")
BEHAVIOR_FEATURE_COLUMNS = (
    "velocity_magnitude_correlation",
    "mean_inter_agent_distance_m",
    "std_inter_agent_distance_m",
    "minimum_inter_agent_distance_m",
    "agent_a_path_length_m",
    "agent_b_path_length_m",
    "agent_a_mean_speed_m_s",
    "agent_a_std_speed_m_s",
    "agent_a_max_speed_m_s",
    "agent_b_mean_speed_m_s",
    "agent_b_std_speed_m_s",
    "agent_b_max_speed_m_s",
    "agent_a_mean_velocity_x_m_s",
    "agent_b_mean_velocity_x_m_s",
)


@dataclass(frozen=True)
class MultimodalCVConfig:
    """Fixed protocol settings for event-level grouped model comparison."""

    n_splits: int = 5
    random_state: int = 17
    target_column: str = "task_state"
    group_column: str = "simulation_seed"

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_splits, bool)
            or not isinstance(self.n_splits, int)
            or self.n_splits < 2
        ):
            raise ValueError("n_splits must be an integer of at least 2")
        if isinstance(self.random_state, bool) or not isinstance(
            self.random_state, int
        ):
            raise TypeError("random_state must be an integer")
        if not self.target_column or not self.group_column:
            raise ValueError("target and group column names must be non-empty")


@dataclass(frozen=True)
class MultimodalComparisonResult:
    """Out-of-fold model predictions, metrics, and evaluation provenance."""

    metrics: dict[str, dict[str, Any]]
    fold_metrics: dict[str, tuple[dict[str, Any], ...]]
    predictions: pd.DataFrame
    class_balance: dict[str, int]
    metadata: dict[str, Any]


def _feature_columns(rows: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    eeg = tuple(
        column
        for column in rows.columns
        if column.startswith(EEG_FEATURE_PREFIXES)
        and pd.api.types.is_numeric_dtype(rows[column])
    )
    behavior = tuple(
        column
        for column in BEHAVIOR_FEATURE_COLUMNS
        if column in rows.columns and pd.api.types.is_numeric_dtype(rows[column])
    )
    if not eeg:
        raise ValueError("unified feature table contains no EEG feature columns")
    if not behavior:
        raise ValueError("unified feature table contains no behavioral feature columns")
    return {"eeg_only": eeg, "behavior_only": behavior, "multimodal": eeg + behavior}


def _scores(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_order": labels,
    }


def evaluate_multimodal_comparison(
    feature_table: UnifiedFeatureTable,
    config: MultimodalCVConfig | None = None,
) -> MultimodalComparisonResult:
    """Compare EEG-only, behavior-only, and fused logistic models using OOF CV.

    All rows sharing a simulation seed are kept together by
    :class:`StratifiedGroupKFold`. A StandardScaler is fitted separately inside
    each training-fold pipeline. Target labels are scripted task-state labels,
    not annotations inferred from EEG. This evaluation only measures behavior
    of the constructed simulated dataset.
    """
    selected = config or MultimodalCVConfig()
    if feature_table.metadata.get("data_source_eeg") != "synthetic":
        raise ValueError("model comparison requires synthetic EEG provenance")
    if feature_table.metadata.get("data_source_behavior") != "simulated":
        raise ValueError("model comparison requires simulated behavior provenance")
    if feature_table.metadata.get("label_source") != "scripted_simulation_ground_truth":
        raise ValueError(
            "target labels must be explicitly sourced from scripted simulation"
        )
    rows = feature_table.rows
    for required in (selected.target_column, selected.group_column):
        if required not in rows:
            raise ValueError(f"feature table is missing required column {required!r}")
    if len(rows) < 1:
        raise ValueError("feature table must contain at least one row")
    if rows[selected.target_column].isna().any():
        raise ValueError("target labels must not be missing")
    targets = rows[selected.target_column].astype(str).to_numpy()
    seeds = rows[selected.group_column]
    sessions = (
        rows["session_id"].astype(str)
        if "session_id" in rows
        else pd.Series("run", index=rows.index)
    )
    groups = (sessions + "::seed=" + seeds.astype(str)).to_numpy()
    labels = sorted(np.unique(targets).tolist())
    if len(labels) < 2:
        raise ValueError("target must contain at least two distinct task-state labels")
    if pd.isna(seeds).any() or pd.isna(sessions).any():
        raise ValueError("group identifiers must not be missing")
    group_counts = {
        label: int(np.unique(groups[targets == label]).size) for label in labels
    }
    if min(group_counts.values()) < selected.n_splits:
        raise ValueError(
            "each class must occur in at least n_splits independent simulation groups"
        )
    modalities = _feature_columns(rows)
    for columns in modalities.values():
        values = rows.loc[:, columns].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("model features must contain only finite values")

    splitter = StratifiedGroupKFold(
        n_splits=selected.n_splits,
        shuffle=True,
        random_state=selected.random_state,
    )
    splits = list(splitter.split(np.zeros(len(rows)), targets, groups=groups))
    for train_index, test_index in splits:
        if set(groups[train_index]) & set(groups[test_index]):
            raise RuntimeError("group leakage detected between train and test rows")
        if set(targets[test_index]) != set(labels):
            raise ValueError(
                "a stratified group fold lacks a class; reduce n_splits or add independent runs"
            )

    predictions: list[dict[str, Any]] = []
    fold_metrics: dict[str, list[dict[str, Any]]] = {name: [] for name in modalities}
    oof_by_model: dict[str, np.ndarray] = {
        name: np.empty(len(rows), dtype=object) for name in modalities
    }
    for fold_index, (train_index, test_index) in enumerate(splits):
        for model_name, columns in modalities.items():
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=selected.random_state,
                ),
            )
            estimator.fit(rows.iloc[train_index].loc[:, columns], targets[train_index])
            predicted = estimator.predict(rows.iloc[test_index].loc[:, columns])
            oof_by_model[model_name][test_index] = predicted
            fold_metrics[model_name].append(
                {
                    "fold": fold_index,
                    "test_groups": sorted(np.unique(groups[test_index]).tolist()),
                    **_scores(targets[test_index], predicted, labels),
                }
            )
            for index, prediction in zip(test_index, predicted, strict=True):
                predictions.append(
                    {
                        "model": model_name,
                        "row_index": int(index),
                        "session_id": str(sessions.iloc[index]),
                        "simulation_seed": str(seeds.iloc[index]),
                        "group": str(groups[index]),
                        "fold": fold_index,
                        "true_label": str(targets[index]),
                        "predicted_label": str(prediction),
                    }
                )

    metric_result: dict[str, dict[str, Any]] = {}
    for model_name, oof_predictions in oof_by_model.items():
        if any(value is None for value in oof_predictions):
            raise RuntimeError("not every row received an out-of-fold prediction")
        scores = _scores(targets, oof_predictions.astype(str), labels)
        fold_balanced = np.asarray(
            [row["balanced_accuracy"] for row in fold_metrics[model_name]], dtype=float
        )
        fold_f1 = np.asarray(
            [row["macro_f1"] for row in fold_metrics[model_name]], dtype=float
        )
        scores["fold_balanced_accuracy_mean"] = float(np.mean(fold_balanced))
        scores["fold_balanced_accuracy_std"] = float(np.std(fold_balanced, ddof=0))
        scores["fold_macro_f1_mean"] = float(np.mean(fold_f1))
        scores["fold_macro_f1_std"] = float(np.std(fold_f1, ddof=0))
        metric_result[model_name] = scores

    return MultimodalComparisonResult(
        metrics=metric_result,
        fold_metrics={name: tuple(values) for name, values in fold_metrics.items()},
        predictions=pd.DataFrame(predictions).sort_values(
            ["model", "row_index"], ignore_index=True
        ),
        class_balance={label: int(np.sum(targets == label)) for label in labels},
        metadata={
            "data_source_eeg": "synthetic",
            "data_source_behavior": "simulated",
            "target": selected.target_column,
            "target_source": "scripted_simulation_ground_truth",
            "target_is_inferred_from_eeg": False,
            "group_definition": "session_id combined with simulation_seed",
            "n_splits": selected.n_splits,
            "random_state": selected.random_state,
            "groups_disjoint_by_fold": True,
            "all_rows_receive_out_of_fold_prediction": True,
            "feature_columns": {name: list(cols) for name, cols in modalities.items()},
            "scaling": "StandardScaler fitted within each training-fold pipeline",
            "classifier": "class-weighted logistic regression",
            "metrics": [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "confusion_matrix",
            ],
            "limitations": [
                "Scores describe prediction of authored labels in constructed synthetic/simulated data.",
                "They are not estimates of human collaboration-state prediction or real-world validity.",
                "Fold scores are descriptive; confidence intervals and hyperparameter tuning are not performed.",
            ],
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
    )
