"""Rule contracts and real PyBullet evaluation for adaptive assistance."""

from __future__ import annotations

import numpy as np
import pytest

from neurosync.simulation import CollaborativeTaskConfig
from neurosync.simulation import task as simulation_task
from neurosync.simulation.assistance import (
    AssistanceEvaluation,
    AssistancePolicyConfig,
    RuleBasedAssistancePolicy,
    evaluate_rule_based_assistance,
)
from neurosync.simulation.task import CollaborativeTaskResult, TaskEvent
from neurosync.visualization.assistance import plot_assistance_evaluation


def _observation(**updates):
    return {
        "time_seconds": 1.0,
        "task_state": "synchronized",
        "human_speed_m_s": 0.1,
        "robot_speed_m_s": 0.1,
        "inter_agent_distance_m": 0.5,
        "human_idle_duration_seconds": 0.0,
        **updates,
    }


def test_policy_is_baseline_when_no_rule_triggers() -> None:
    action = RuleBasedAssistancePolicy().decide(_observation())
    assert action["assistance_level"] == "baseline"
    assert action["reason"] == "no_rule_triggered"
    assert action["robot_target_delay_fraction"] == pytest.approx(0.55)
    assert action["robot_max_force_n"] == pytest.approx(8.0)


def test_policy_elevates_support_for_desynchronization_or_hesitation() -> None:
    policy = RuleBasedAssistancePolicy()
    delayed = policy.decide(_observation(task_state="desynchronized"))
    hesitant = policy.decide(
        _observation(human_speed_m_s=0.0, human_idle_duration_seconds=0.4)
    )
    for action in (delayed, hesitant):
        assert action["assistance_level"] == "elevated"
        assert action["robot_target_delay_fraction"] < 0.55
        assert action["robot_max_force_n"] > 8.0
        assert action["reason"] != "no_rule_triggered"


def test_policy_rejects_invalid_observations_and_configuration() -> None:
    with pytest.raises(ValueError, match="task state"):
        RuleBasedAssistancePolicy().decide(_observation(task_state="uncertain"))
    with pytest.raises(ValueError, match="hesitation_duration"):
        AssistancePolicyConfig(hesitation_duration_seconds=0)


def test_assistance_plot_is_labeled_as_simulation_only() -> None:
    times = np.array([0.0, 1.0, 2.0])
    positions_a = np.column_stack((times, np.zeros(3), np.ones(3)))
    positions_b = positions_a + np.array([0.2, 0.0, 0.0])
    velocities = np.zeros((3, 3))
    speeds = np.zeros(3)
    event = TaskEvent(0, "handover", "desynchronized", 0.0, 2.0)

    def run(assisted: bool) -> CollaborativeTaskResult:
        decisions = (
            ({"time_seconds": 1.0, "assistance_level": "elevated"},) if assisted else ()
        )
        return CollaborativeTaskResult(
            time_seconds=times,
            agent_a_position_m=positions_a,
            agent_b_position_m=positions_b,
            agent_a_velocity_m_s=velocities,
            agent_b_velocity_m_s=velocities,
            agent_a_speed_m_s=speeds,
            agent_b_speed_m_s=speeds,
            inter_agent_distance_m=np.full(3, 0.2),
            events=(event,),
            velocity_correlation=0.0,
            idle_duration_seconds={"agent_a": 0.0, "agent_b": 0.0},
            metadata={"data_source": "simulated"},
            assistance_decisions=decisions,
        )

    evaluation = AssistanceEvaluation(
        unassisted=run(False),
        adaptive=run(True),
        metrics={},
        metadata={"data_source": "simulated"},
    )
    figure = plot_assistance_evaluation(evaluation)
    assert "not evidence of human benefit" in figure._suptitle.get_text()
    import matplotlib.pyplot as plt

    plt.close(figure)


@pytest.mark.skipif(
    simulation_task.bullet is None,
    reason="paired policy evaluation requires real PyBullet",
)
def test_policy_changes_actual_robot_proxy_behavior_and_logs_decisions() -> None:
    result = evaluate_rule_based_assistance(
        CollaborativeTaskConfig(sampling_frequency_hz=20, seed=83)
    )
    baseline = result.unassisted
    adaptive = result.adaptive
    assert np.any(
        np.linalg.norm(np.diff(baseline.agent_b_position_m, axis=0), axis=1) > 0
    )
    assert np.any(
        np.linalg.norm(np.diff(adaptive.agent_b_position_m, axis=0), axis=1) > 0
    )
    assert not np.array_equal(baseline.agent_b_position_m, adaptive.agent_b_position_m)
    assert baseline.assistance_decisions == ()
    assert len(adaptive.assistance_decisions) == adaptive.time_seconds.size - 1
    assert any(
        item["assistance_level"] == "elevated" for item in adaptive.assistance_decisions
    )
    assert adaptive.metadata["assistance_policy"]["enabled"] is True
    assert set(result.metrics) == {"unassisted", "adaptive"}
    assert result.metadata["outcomes_are_human_validity_evidence"] is False
