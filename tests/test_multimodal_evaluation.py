"""Tests for leakage-resistant modality comparisons on fixture rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurosync.features.unified import UnifiedFeatureTable
from neurosync.modeling.multimodal import (
    MultimodalCVConfig,
    evaluate_multimodal_comparison,
)
from neurosync.visualization.model_comparison import plot_model_comparison


def _feature_table(*, provenance: str = "synthetic") -> UnifiedFeatureTable:
    rows: list[dict[str, object]] = []
    for seed in range(20):
        session = f"session-{seed}"
        for event_index, label in enumerate(
            ("synchronized", "desynchronized", "synchronized")
        ):
            state_value = float(label == "desynchronized")
            noise = (seed % 5) * 0.01
            rows.append(
                {
                    "session_id": session,
                    "simulation_seed": seed,
                    "task_state": label,
                    "eeg_a_mean_alpha_relative_power": state_value + noise,
                    "eeg_b_mean_beta_relative_power": state_value - noise,
                    "eeg_mean_alpha_coherence": state_value + event_index * 0.01,
                    "agent_a_mean_speed_m_s": 0.1 + state_value + noise,
                    "agent_b_max_speed_m_s": 0.2 + state_value - noise,
                    "mean_inter_agent_distance_m": 0.5 - 0.1 * state_value,
                    "velocity_magnitude_correlation": 0.4 + 0.2 * state_value,
                }
            )
    return UnifiedFeatureTable(
        rows=pd.DataFrame(rows),
        metadata={
            "data_source_eeg": provenance,
            "data_source_behavior": "simulated",
            "label_source": "scripted_simulation_ground_truth",
        },
    )


def test_comparison_reports_oof_metrics_and_keeps_runs_grouped() -> None:
    result = evaluate_multimodal_comparison(
        _feature_table(), MultimodalCVConfig(n_splits=5, random_state=7)
    )
    assert set(result.metrics) == {"eeg_only", "behavior_only", "multimodal"}
    assert result.class_balance == {"desynchronized": 20, "synchronized": 40}
    assert len(result.predictions) == 3 * 60
    assert set(result.predictions["true_label"]) == {
        "synchronized",
        "desynchronized",
    }
    for model, metrics in result.metrics.items():
        assert 0 <= metrics["balanced_accuracy"] <= 1
        assert 0 <= metrics["macro_f1"] <= 1
        assert np.asarray(metrics["confusion_matrix"]).shape == (2, 2)
        assert len(result.fold_metrics[model]) == 5
        group_fold_counts = (
            result.predictions.query("model == @model")
            .groupby("group")["fold"]
            .nunique()
        )
        assert (group_fold_counts == 1).all()
    assert result.metadata["groups_disjoint_by_fold"] is True
    assert result.metadata["target_is_inferred_from_eeg"] is False


def test_comparison_is_deterministic() -> None:
    first = evaluate_multimodal_comparison(_feature_table())
    second = evaluate_multimodal_comparison(_feature_table())
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    assert first.metrics == second.metrics


def test_comparison_rejects_bad_provenance_and_insufficient_groups() -> None:
    with pytest.raises(ValueError, match="synthetic"):
        evaluate_multimodal_comparison(_feature_table(provenance="human"))
    table = _feature_table()
    small = UnifiedFeatureTable(
        rows=table.rows.iloc[:6].copy(), metadata=table.metadata
    )
    with pytest.raises(ValueError, match="independent simulation groups"):
        evaluate_multimodal_comparison(small, MultimodalCVConfig(n_splits=5))


def test_comparison_rejects_missing_or_nonfinite_features() -> None:
    table = _feature_table()
    bad_rows = table.rows.copy()
    bad_rows.loc[0, "eeg_a_mean_alpha_relative_power"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate_multimodal_comparison(
            UnifiedFeatureTable(rows=bad_rows, metadata=table.metadata)
        )
    bad_rows = table.rows.copy()
    bad_rows.loc[0, "task_state"] = None
    with pytest.raises(ValueError, match="target labels"):
        evaluate_multimodal_comparison(
            UnifiedFeatureTable(rows=bad_rows, metadata=table.metadata)
        )


def test_comparison_figure_labels_synthetic_target_and_modalities() -> None:
    result = evaluate_multimodal_comparison(_feature_table())
    figure = plot_model_comparison(result)
    assert len(figure.axes) == 5
    assert "synthetic/simulated" in figure._suptitle.get_text()
    assert "not human-state validity" in figure._suptitle.get_text()
    assert [axis.get_title() for axis in figure.axes[1:4]] == [
        "EEG only",
        "Behavior only",
        "EEG + behavior",
    ]
    import matplotlib.pyplot as plt

    plt.close(figure)
