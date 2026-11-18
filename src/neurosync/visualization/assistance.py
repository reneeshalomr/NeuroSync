"""Plots for simulation-only adaptive assistance inspection."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from neurosync.simulation.assistance import AssistanceEvaluation


def plot_assistance_evaluation(result: AssistanceEvaluation) -> Figure:
    """Compare proxy trajectories, separation, and logged rule decisions."""
    if result.metadata.get("data_source") != "simulated":
        raise ValueError("assistance plots require simulated evaluation provenance")
    if not result.adaptive.assistance_decisions:
        raise ValueError("adaptive result must contain policy decision logs")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for run, style, label in (
        (result.unassisted, "--", "Unassisted"),
        (result.adaptive, "-", "Rule-based assistance"),
    ):
        axes[0].plot(
            run.agent_a_position_m[:, 0],
            run.agent_a_position_m[:, 2],
            linestyle=style,
            label=f"Agent A ({label})",
            alpha=0.8,
        )
        axes[0].plot(
            run.agent_b_position_m[:, 0],
            run.agent_b_position_m[:, 2],
            linestyle=style,
            label=f"Robot proxy ({label})",
            alpha=0.8,
        )
    axes[0].set_xlabel("Workspace x (m)")
    axes[0].set_ylabel("Workspace z (m)")
    axes[0].set_title("PyBullet proxy trajectories")
    axes[0].legend(fontsize="small")
    for run, label in (
        (result.unassisted, "Unassisted"),
        (result.adaptive, "Rule-based assistance"),
    ):
        axes[1].plot(run.time_seconds, run.inter_agent_distance_m, label=label)
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].set_ylabel("Inter-agent distance (m)")
    axes[1].set_title("Simulated workspace separation")
    axes[1].legend(fontsize="small")

    decisions = result.adaptive.assistance_decisions
    decision_times = np.asarray([row["time_seconds"] for row in decisions])
    levels = np.asarray(
        [1 if row["assistance_level"] == "elevated" else 0 for row in decisions]
    )
    axes[2].step(decision_times, levels, where="post", color="tab:orange")
    axes[2].set_yticks([0, 1], ["Baseline", "Elevated"])
    axes[2].set_xlabel("Simulation time (s)")
    axes[2].set_title("Logged policy action")
    axes[2].grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Simulation-only assistance comparison\n"
        "Scripted dynamic proxies; not evidence of human benefit or real-robot safety"
    )
    return figure
