"""Publication-style plots for synthetic grouped modality comparisons."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from neurosync.modeling.multimodal import MultimodalComparisonResult

MODEL_ORDER = ("eeg_only", "behavior_only", "multimodal")
MODEL_LABELS = {
    "eeg_only": "EEG only",
    "behavior_only": "Behavior only",
    "multimodal": "EEG + behavior",
}


def plot_model_comparison(result: MultimodalComparisonResult) -> Figure:
    """Plot out-of-fold balanced accuracy/F1 and confusion matrices.

    The figure explicitly describes target labels as synthetic simulation
    ground truth. It is suitable for reporting this software experiment, not
    for claims about human-state prediction.
    """
    missing = set(MODEL_ORDER) - set(result.metrics)
    if missing:
        raise ValueError(f"comparison result is missing models: {sorted(missing)}")
    labels = result.metrics[MODEL_ORDER[0]].get("class_order")
    if not isinstance(labels, list) or not labels:
        raise ValueError("model metrics must provide a non-empty class_order")
    for model in MODEL_ORDER:
        matrix = np.asarray(result.metrics[model].get("confusion_matrix"))
        if matrix.shape != (len(labels), len(labels)) or not np.isfinite(matrix).all():
            raise ValueError(f"model {model!r} has an invalid confusion matrix")

    figure = plt.figure(figsize=(12, 7), layout="constrained")
    grid = figure.add_gridspec(2, 3, height_ratios=(0.8, 1.2))
    metrics_ax = figure.add_subplot(grid[0, :])
    metric_names = ("balanced_accuracy", "macro_f1")
    positions = np.arange(len(MODEL_ORDER))
    bar_width = 0.34
    for metric_index, metric in enumerate(metric_names):
        values = [float(result.metrics[name][metric]) for name in MODEL_ORDER]
        offset = (metric_index - 0.5) * bar_width
        metrics_ax.bar(
            positions + offset, values, width=bar_width, label=metric.replace("_", " ")
        )
    metrics_ax.set_xticks(positions, [MODEL_LABELS[name] for name in MODEL_ORDER])
    metrics_ax.set_ylim(0, 1.05)
    metrics_ax.set_ylabel("Out-of-fold score")
    metrics_ax.set_title("Grouped cross-validation metrics")
    metrics_ax.legend(frameon=False, ncols=2)
    metrics_ax.grid(axis="y", alpha=0.2)

    last_image = None
    for index, model in enumerate(MODEL_ORDER):
        ax = figure.add_subplot(grid[1, index])
        matrix = np.asarray(result.metrics[model]["confusion_matrix"], dtype=float)
        last_image = ax.imshow(matrix, cmap="Blues", vmin=0)
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("Predicted scripted state")
        ax.set_ylabel("True scripted state")
        ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(labels)), labels)
        threshold = float(matrix.max()) / 2 if matrix.size else 0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                ax.text(
                    column,
                    row,
                    f"{int(matrix[row, column])}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                )
    if last_image is not None:
        figure.colorbar(
            last_image, ax=figure.axes[-3:], shrink=0.8, label="OOF row count"
        )
    figure.suptitle(
        "NeuroSync synthetic/simulated task-state evaluation\n"
        "Target: scripted simulation ground truth; not human-state validity"
    )
    return figure
