"""Inspectable rule-based assistance policy and paired simulation evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from neurosync.simulation.task import (
    CollaborativeTaskConfig,
    CollaborativeTaskResult,
    run_collaborative_task,
)


@dataclass(frozen=True)
class AssistancePolicyConfig:
    """Transparent trigger and controller settings for the proxy task."""

    human_idle_threshold_m_s: float = 0.015
    hesitation_duration_seconds: float = 0.30
    desynchronized_delay_fraction: float = 0.20
    robot_kp_per_s2: float = 54.0
    robot_kd_per_s: float = 12.0
    robot_max_force_n: float = 12.0

    def __post_init__(self) -> None:
        for name in (
            "human_idle_threshold_m_s",
            "hesitation_duration_seconds",
            "desynchronized_delay_fraction",
            "robot_kp_per_s2",
            "robot_kd_per_s",
            "robot_max_force_n",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.hesitation_duration_seconds == 0:
            raise ValueError("hesitation_duration_seconds must be positive")
        if self.desynchronized_delay_fraction >= 1:
            raise ValueError("desynchronized_delay_fraction must be below 1")
        if self.robot_kp_per_s2 == 0 or self.robot_max_force_n == 0:
            raise ValueError("robot controller gain and force must be positive")


class RuleBasedAssistancePolicy:
    """Increase robot-proxy tracking effort during scripted delay/hesitation.

    Trigger conditions use measured simulation kinematics and an authored task
    state. This is an interpretable software policy, not a safety controller or
    a policy validated with people or a physical robot.
    """

    policy_name = "rule_based_handover_assistance_v1"

    def __init__(
        self,
        config: AssistancePolicyConfig | None = None,
        *,
        baseline_delay_fraction: float = 0.55,
    ) -> None:
        if (
            isinstance(baseline_delay_fraction, bool)
            or not math.isfinite(baseline_delay_fraction)
            or not 0 <= baseline_delay_fraction < 1
        ):
            raise ValueError("baseline_delay_fraction must be in [0, 1)")
        self.config = config or AssistancePolicyConfig()
        self.baseline_delay_fraction = baseline_delay_fraction

    def decide(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Return a fully logged robot control action for one simulation step."""
        task_state = observation.get("task_state")
        human_speed = float(observation.get("human_speed_m_s", math.nan))
        idle_duration = float(observation.get("human_idle_duration_seconds", math.nan))
        if task_state not in {"synchronized", "desynchronized"}:
            raise ValueError("policy observation contains an unsupported task state")
        if not math.isfinite(human_speed) or human_speed < 0:
            raise ValueError("human_speed_m_s must be finite and non-negative")
        if not math.isfinite(idle_duration) or idle_duration < 0:
            raise ValueError(
                "human_idle_duration_seconds must be finite and non-negative"
            )
        if task_state == "desynchronized":
            reason = "scripted_task_state_desynchronized"
        elif (
            human_speed < self.config.human_idle_threshold_m_s
            and idle_duration >= self.config.hesitation_duration_seconds
        ):
            reason = "proxy_hesitation_duration_threshold"
        else:
            reason = "no_rule_triggered"
        assisted = reason != "no_rule_triggered"
        return {
            "policy_name": self.policy_name,
            "assistance_level": "elevated" if assisted else "baseline",
            "reason": reason,
            "robot_target_delay_fraction": (
                self.config.desynchronized_delay_fraction
                if assisted
                else self.baseline_delay_fraction
            ),
            "robot_kp_per_s2": (self.config.robot_kp_per_s2 if assisted else 36.0),
            "robot_kd_per_s": self.config.robot_kd_per_s if assisted else 12.0,
            "robot_max_force_n": (self.config.robot_max_force_n if assisted else 8.0),
        }


@dataclass(frozen=True)
class AssistanceEvaluation:
    """Paired unassisted/adaptive runs and descriptive task metrics."""

    unassisted: CollaborativeTaskResult
    adaptive: CollaborativeTaskResult
    metrics: dict[str, dict[str, float | int]]
    metadata: dict[str, Any]


def _run_metrics(result: CollaborativeTaskResult) -> dict[str, float | int]:
    handover = next(
        (event for event in result.events if event.phase_name == "handover"), None
    )
    if handover is None:
        raise ValueError("assistance evaluation requires a handover phase")
    event_mask = (result.time_seconds >= handover.start_time_seconds) & (
        result.time_seconds < handover.end_time_seconds
    )
    if not event_mask.any():
        raise ValueError("handover phase contains no simulation samples")
    handover_distance = result.inter_agent_distance_m[event_mask]
    robot_speed = result.agent_b_speed_m_s
    path_length = float(
        np.linalg.norm(np.diff(result.agent_b_position_m, axis=0), axis=1).sum()
    )
    elevated = sum(
        decision.get("assistance_level") == "elevated"
        for decision in result.assistance_decisions
    )
    return {
        "handover_mean_inter_agent_distance_m": float(np.mean(handover_distance)),
        "handover_min_inter_agent_distance_m": float(np.min(handover_distance)),
        "robot_mean_speed_m_s": float(np.mean(robot_speed)),
        "robot_path_length_m": path_length,
        "elevated_decision_count": int(elevated),
        "decision_count": len(result.assistance_decisions),
    }


def evaluate_rule_based_assistance(
    task_config: CollaborativeTaskConfig | None = None,
    policy: RuleBasedAssistancePolicy | None = None,
) -> AssistanceEvaluation:
    """Run matched unassisted and policy-controlled PyBullet task conditions.

    The metrics are descriptive simulation outcomes. The function does not
    select a preferred policy, estimate human benefit, or establish safety.
    """
    config = task_config or CollaborativeTaskConfig()
    selected_policy = policy or RuleBasedAssistancePolicy(
        baseline_delay_fraction=config.desynchronization_delay_fraction
    )
    unassisted = run_collaborative_task(config)
    adaptive = run_collaborative_task(config, assistance_policy=selected_policy)
    return AssistanceEvaluation(
        unassisted=unassisted,
        adaptive=adaptive,
        metrics={
            "unassisted": _run_metrics(unassisted),
            "adaptive": _run_metrics(adaptive),
        },
        metadata={
            "data_source": "simulated",
            "engine": adaptive.metadata.get("engine"),
            "simulation_seed": config.seed,
            "policy_name": selected_policy.policy_name,
            "comparison": "same authored task schedule and seed, policy disabled versus enabled",
            "outcomes_are_human_validity_evidence": False,
            "interpretation": "descriptive controller behavior in a scripted simulation only",
            "metrics_are_success_claims": False,
            "limitations": [
                "Human agent is a scripted dynamic proxy, not a participant.",
                "The rule trigger uses simulation ground-truth task phase and proxy kinematics.",
                "One paired seed is an executable demonstration, not a statistical policy study.",
                "No robot safety, human benefit, or deployment validation is performed.",
            ],
        },
    )
