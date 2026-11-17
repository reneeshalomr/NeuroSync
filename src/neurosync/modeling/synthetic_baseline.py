"""Grouped baseline classification for the generator's synthetic condition.

The target is the software generator setting HIGH_SHARED_COMPONENT versus
LOW_SHARED_COMPONENT. It is not a human collaboration-state annotation.
Features and scores from this controlled fixture are workflow checks only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy
import sklearn
from numpy.typing import NDArray
from sklearn.dummy import DummyClassifier
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

from neurosync.collaboration.synchrony import compute_synthetic_coherence
from neurosync.data.synthetic_eeg import (
    EEG_FREQUENCY_BANDS_HZ,
    ExperimentalCondition,
    generate_synthetic_eeg,
)
from neurosync.features.spectral import extract_spectral_features
from neurosync.preprocessing.raw import synthetic_participant_to_raw

CLASS_ORDER = (
    ExperimentalCondition.LOW_SHARED_COMPONENT.value,
    ExperimentalCondition.HIGH_SHARED_COMPONENT.value,
)


@dataclass(frozen=True)
class SyntheticConditionDataset:
    """Paired synthetic-condition rows with seed-pair groups and provenance."""

    features: NDArray[np.float64]
    labels: NDArray[np.str_]
    groups: NDArray[np.str_]
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SyntheticBaselineEvaluation:
    """Held-out grouped metrics and the experiment protocol metadata."""

    class_balance: dict[str, int]
    train_groups: tuple[str, ...]
    test_groups: tuple[str, ...]
    metrics: dict[str, dict[str, Any]]
    metadata: dict[str, Any]


def build_synthetic_condition_dataset(
    n_sessions: int = 16,
    *,
    seed: int = 4200,
    sampling_frequency_hz: float = 256.0,
    duration_seconds: float = 8.0,
    n_channels: int = 4,
) -> SyntheticConditionDataset:
    """Build paired high/low generator-setting samples, grouped by seed.

    Each seed generates one example under each condition. Both examples share
    a group identifier and must therefore remain on the same side of any split.
    Features pool per-channel relative spectral powers and entropy across the
    two simulated participants, plus their matched-channel coherence summaries.
    """
    if (
        isinstance(n_sessions, bool)
        or not isinstance(n_sessions, int)
        or n_sessions < 4
    ):
        raise ValueError("n_sessions must be an integer of at least 4")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if (
        isinstance(n_channels, bool)
        or not isinstance(n_channels, int)
        or n_channels < 1
    ):
        raise ValueError("n_channels must be a positive integer")
    if not math.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 90.7:
        raise ValueError(
            "sampling_frequency_hz must exceed the generator's 90.7 Hz Nyquist limit"
        )
    if not math.isfinite(duration_seconds) or duration_seconds < 2.0:
        raise ValueError(
            "duration_seconds must be at least 2 seconds for default feature windows"
        )

    band_names = tuple(EEG_FREQUENCY_BANDS_HZ)
    feature_names = (
        tuple(
            name
            for band in band_names
            for name in (f"relative_{band}_mean", f"relative_{band}_std")
        )
        + (
            "spectral_entropy_mean",
            "spectral_entropy_std",
            "normalized_entropy_mean",
            "normalized_entropy_std",
        )
        + tuple(f"coherence_{band}_mean" for band in band_names)
    )

    rows: list[list[float]] = []
    labels: list[str] = []
    groups: list[str] = []
    conditions = (
        ExperimentalCondition.LOW_SHARED_COMPONENT,
        ExperimentalCondition.HIGH_SHARED_COMPONENT,
    )
    for session_index in range(n_sessions):
        session_seed = seed + session_index
        group = f"paired_seed_{session_index:04d}"
        for condition in conditions:
            data = generate_synthetic_eeg(
                sampling_frequency_hz=sampling_frequency_hz,
                duration_seconds=duration_seconds,
                n_channels=n_channels,
                seed=session_seed,
                condition=condition,
            )
            raw_a = synthetic_participant_to_raw(data, "A")
            raw_b = synthetic_participant_to_raw(data, "B")
            spectral = [
                extract_spectral_features(raw_a),
                extract_spectral_features(raw_b),
            ]
            pooled_channels = [
                channel for result in spectral for channel in result.channel_features
            ]
            row: list[float] = []
            for band in band_names:
                values = np.asarray(
                    [channel.relative_band_power[band] for channel in pooled_channels]
                )
                row.extend((float(values.mean()), float(values.std(ddof=0))))
            entropy = np.asarray(
                [channel.spectral_entropy_nats for channel in pooled_channels]
            )
            normalized_entropy = np.asarray(
                [channel.normalized_spectral_entropy for channel in pooled_channels]
            )
            row.extend((float(entropy.mean()), float(entropy.std(ddof=0))))
            row.extend(
                (
                    float(normalized_entropy.mean()),
                    float(normalized_entropy.std(ddof=0)),
                )
            )
            coherence_result = compute_synthetic_coherence(raw_a, raw_b)
            row.extend(
                coherence_result.band_mean_coherence[band] for band in band_names
            )
            rows.append(row)
            labels.append(condition.value)
            groups.append(group)

    feature_matrix = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(feature_matrix).all():
        raise ValueError("feature extraction produced non-finite values")
    return SyntheticConditionDataset(
        features=feature_matrix,
        labels=np.asarray(labels, dtype=np.str_),
        groups=np.asarray(groups, dtype=np.str_),
        feature_names=feature_names,
        metadata={
            "data_source": "synthetic",
            "target_name": "generator_shared_component_condition",
            "target_definition": "HIGH_SHARED_COMPONENT vs LOW_SHARED_COMPONENT generator settings",
            "not_human_collaboration_state": True,
            "n_sessions": n_sessions,
            "rows_per_session": 2,
            "paired_conditions_share_seed_and_group": True,
            "base_seed": seed,
            "sampling_frequency_hz": sampling_frequency_hz,
            "duration_seconds": duration_seconds,
            "n_channels_per_participant": n_channels,
            "feature_source": "relative Welch band power, spectral entropy, and descriptive coherence",
            "feature_names": list(feature_names),
            "signal_unit": "V",
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
    )


def _wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Wilson score interval for a binomial rate over independent seed groups."""
    if total < 1 or not 0 <= successes <= total:
        raise ValueError(
            "Wilson interval requires 0 <= successes <= total and total > 0"
        )
    rate = successes / total
    z_squared = z**2
    denominator = 1 + z_squared / total
    center = (rate + z_squared / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1 - rate) / total + z_squared / (4 * total**2))
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - margin)
    upper = 1.0 if successes == total else min(1.0, center + margin)
    return lower, upper


def evaluate_synthetic_baselines(
    dataset: SyntheticConditionDataset,
    *,
    random_state: int = 17,
    n_splits: int = 4,
) -> SyntheticBaselineEvaluation:
    """Evaluate majority and standardized logistic baselines on held-out groups.

    Uses the first deterministic fold from shuffled StratifiedGroupKFold. The
    scaler is fitted inside a scikit-learn Pipeline on training examples only.
    A Wilson interval summarizes the held-out fraction of seed groups for which
    both paired condition rows are classified correctly. Its sampling unit is
    the seed group, and it does not support human-population generalization.
    """
    if dataset.metadata.get("data_source") != "synthetic":
        raise ValueError("dataset must carry synthetic provenance")
    if dataset.features.ndim != 2 or dataset.features.shape[0] != dataset.labels.size:
        raise ValueError("features must be 2D with one row per label")
    if (
        dataset.groups.shape != dataset.labels.shape
        or not np.isfinite(dataset.features).all()
    ):
        raise ValueError("groups and labels must align and features must be finite")
    if set(np.unique(dataset.labels)) != set(CLASS_ORDER):
        raise ValueError(
            "dataset must contain both expected generator-condition labels"
        )
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")
    if np.unique(dataset.groups).size < n_splits:
        raise ValueError("n_splits cannot exceed the number of independent seed groups")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    train_index, test_index = next(
        splitter.split(dataset.features, dataset.labels, groups=dataset.groups)
    )
    train_groups = tuple(sorted(np.unique(dataset.groups[train_index]).tolist()))
    test_groups = tuple(sorted(np.unique(dataset.groups[test_index]).tolist()))
    if set(train_groups) & set(test_groups):
        raise RuntimeError(
            "seed-pair group leakage detected across the evaluation split"
        )
    x_train, x_test = dataset.features[train_index], dataset.features[test_index]
    y_train, y_test = dataset.labels[train_index], dataset.labels[test_index]
    models = {
        "majority_dummy": DummyClassifier(strategy="most_frequent"),
        "standardized_logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, class_weight=None, random_state=random_state
            ),
        ),
    }
    results: dict[str, dict[str, Any]] = {}
    test_groups_array = dataset.groups[test_index]
    for name, estimator in models.items():
        estimator.fit(x_train, y_train)
        predicted = estimator.predict(x_test)
        group_successes = sum(
            bool(
                np.all(
                    predicted[test_groups_array == group]
                    == y_test[test_groups_array == group]
                )
            )
            for group in test_groups
        )
        group_interval = _wilson_interval(group_successes, len(test_groups))
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predicted)),
            "macro_f1": float(
                f1_score(
                    y_test,
                    predicted,
                    labels=CLASS_ORDER,
                    average="macro",
                    zero_division=0,
                )
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predicted, labels=CLASS_ORDER
            ).tolist(),
            "test_support": {
                label: int(np.count_nonzero(y_test == label)) for label in CLASS_ORDER
            },
            "paired_group_exact_accuracy": group_successes / len(test_groups),
            "paired_group_exact_accuracy_95pct_wilson_interval": list(group_interval),
        }
    balance = {
        label: int(np.count_nonzero(dataset.labels == label)) for label in CLASS_ORDER
    }
    return SyntheticBaselineEvaluation(
        class_balance=balance,
        train_groups=train_groups,
        test_groups=test_groups,
        metrics=results,
        metadata={
            "data_source": "synthetic",
            "target_name": dataset.metadata["target_name"],
            "target_definition": dataset.metadata["target_definition"],
            "not_human_collaboration_state": True,
            "splitter": "StratifiedGroupKFold; first fold only",
            "n_splits": n_splits,
            "split_random_state": random_state,
            "grouping": "paired synthetic condition rows sharing one generator seed",
            "scaler_fit_scope": "training fold only through Pipeline",
            "uncertainty": "95% Wilson score interval for exact paired-group accuracy; seed groups are sampling units",
            "feature_names": list(dataset.feature_names),
            "models": list(models),
            "limitations": "synthetic generator setting only; not evidence of human-state classification or generalization",
            "scikit_learn_version": sklearn.__version__,
        },
    )
