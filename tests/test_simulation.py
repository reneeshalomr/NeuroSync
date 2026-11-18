"""Tests for the scripted shared-workspace task simulation contract."""

from __future__ import annotations

import json

import numpy as np
import pytest

from neurosync.simulation import (
    CollaborativeTaskConfig,
    run_collaborative_task,
    save_simulation_result,
)
from neurosync.simulation import task as simulation_task
from neurosync.simulation.assistance import RuleBasedAssistancePolicy


class _FakeBullet:
    DIRECT = 0
    GEOM_BOX = 1
    GEOM_SPHERE = 2
    WORLD_FRAME = 0

    def __init__(self) -> None:
        self.next_id = 0
        self.bodies = {}
        self.gravity = np.zeros(3)
        self.dt = 0.05

    def connect(self, mode):
        return 1

    def resetSimulation(self, **kwargs):
        pass

    def setGravity(self, *args, **kwargs):
        self.gravity = np.asarray(args[:3], dtype=float)

    def setTimeStep(self, *args, **kwargs):
        self.dt = float(args[0])

    def createCollisionShape(self, *args, **kwargs):
        self.next_id += 1
        return self.next_id

    def createMultiBody(self, *args, **kwargs):
        self.next_id += 1
        self.bodies[self.next_id] = {
            "mass": float(kwargs["baseMass"]),
            "position": np.asarray(kwargs["basePosition"], dtype=float),
            "velocity": np.zeros(3),
            "force": np.zeros(3),
        }
        return self.next_id

    def getBasePositionAndOrientation(self, body, **kwargs):
        return self.bodies[body]["position"].tolist(), [0.0, 0.0, 0.0, 1.0]

    def getBaseVelocity(self, body, **kwargs):
        return self.bodies[body]["velocity"].tolist(), [0.0, 0.0, 0.0]

    def applyExternalForce(self, body, _link, force, _position, _frame, **kwargs):
        self.bodies[body]["force"] += np.asarray(force)

    def stepSimulation(self, **kwargs):
        for body in self.bodies.values():
            if body["mass"]:
                acceleration = body["force"] / body["mass"] + self.gravity
                body["velocity"] += acceleration * self.dt
                body["position"] += body["velocity"] * self.dt
            body["force"][:] = 0

    def disconnect(self, **kwargs):
        pass


@pytest.fixture
def fake_bullet(monkeypatch):
    engine = _FakeBullet()
    monkeypatch.setattr(simulation_task, "bullet", engine)
    return engine


def test_simulation_telemetry_events_and_metadata(fake_bullet) -> None:
    result = run_collaborative_task(
        CollaborativeTaskConfig(sampling_frequency_hz=10, seed=8)
    )
    assert result.time_seconds.shape == (61,)
    assert result.agent_a_position_m.shape == (61, 3)
    assert result.agent_b_velocity_m_s.shape == (61, 3)
    assert result.inter_agent_distance_m.shape == (61,)
    assert np.isfinite(result.agent_a_speed_m_s).all()
    assert np.isfinite(result.inter_agent_distance_m).all()
    assert -1 <= result.velocity_correlation <= 1
    assert set(result.idle_duration_seconds) == {"agent_a", "agent_b"}
    assert [event.task_state for event in result.events] == [
        "synchronized",
        "desynchronized",
        "synchronized",
    ]
    assert result.events[0].start_time_seconds == 0
    assert result.events[-1].end_time_seconds == pytest.approx(6.0)
    assert result.metadata["data_source"] == "simulated"
    assert (
        result.metadata["task_state_ground_truth_source"] == "scripted phase schedule"
    )


def test_simulation_repeats_and_phase_control_changes_telemetry(fake_bullet) -> None:
    first = run_collaborative_task(
        CollaborativeTaskConfig(sampling_frequency_hz=10, seed=3)
    )
    second = run_collaborative_task(
        CollaborativeTaskConfig(sampling_frequency_hz=10, seed=3)
    )
    np.testing.assert_array_equal(first.agent_a_position_m, second.agent_a_position_m)
    np.testing.assert_array_equal(
        first.inter_agent_distance_m, second.inter_agent_distance_m
    )
    config = CollaborativeTaskConfig(
        sampling_frequency_hz=10,
        phases=(("handover", "synchronized", 6.0),),
    )
    synchronized = run_collaborative_task(config)
    assert [event.phase_name for event in synchronized.events] == ["handover"]
    assert [event.phase_name for event in first.events] != [
        event.phase_name for event in synchronized.events
    ]


def test_assistance_policy_inputs_and_decisions_are_retained(fake_bullet) -> None:
    result = run_collaborative_task(
        CollaborativeTaskConfig(sampling_frequency_hz=10, seed=15),
        assistance_policy=RuleBasedAssistancePolicy(),
    )
    assert result.metadata["assistance_policy"]["enabled"] is True
    assert result.metadata["assistance_policy"]["decision_count"] == 60
    assert len(result.assistance_decisions) == 60
    first = result.assistance_decisions[0]
    assert {
        "task_state",
        "human_speed_m_s",
        "robot_speed_m_s",
        "inter_agent_distance_m",
        "reason",
        "robot_kp_per_s2",
        "assistance_level",
    } <= first.keys()
    assert any(
        decision["assistance_level"] == "elevated"
        for decision in result.assistance_decisions
    )


def test_npz_roundtrip_uses_pickle_free_metadata(fake_bullet, tmp_path) -> None:
    result = run_collaborative_task(CollaborativeTaskConfig(sampling_frequency_hz=10))
    target = save_simulation_result(result, tmp_path / "handover_simulated_run.npz")
    with np.load(target, allow_pickle=False) as archive:
        assert archive["agent_a_position_m"].shape == (61, 3)
        events = json.loads(str(archive["events_json"]))
        metadata = json.loads(str(archive["metadata_json"]))
        assert len(events) == len(result.events)
        assert metadata["data_source"] == "simulated"
        assert json.loads(str(archive["assistance_decisions_json"])) == []


def test_rejects_invalid_simulation_configuration() -> None:
    with pytest.raises(ValueError, match="sampling_frequency"):
        CollaborativeTaskConfig(sampling_frequency_hz=0)
    with pytest.raises(TypeError, match="seed"):
        CollaborativeTaskConfig(seed=True)
    with pytest.raises(ValueError, match="state"):
        CollaborativeTaskConfig(phases=(("phase", "uncertain", 1.0),))
    with pytest.raises(ValueError, match="align"):
        run_collaborative_task(
            CollaborativeTaskConfig(
                sampling_frequency_hz=10, phases=(("approach", "synchronized", 1.01),)
            )
        )


def test_actual_pybullet_smoke_when_optional_dependency_is_installed() -> None:
    if simulation_task.bullet is None:
        pytest.skip("PyBullet optional extra is not installed in this interpreter")
    result = run_collaborative_task(CollaborativeTaskConfig(sampling_frequency_hz=10))
    assert result.metadata["engine"] == "PyBullet DIRECT"
    assert result.time_seconds.size == 61
    assert np.any(
        np.linalg.norm(np.diff(result.agent_a_position_m, axis=0), axis=1) > 0
    )
    assert np.any(
        np.linalg.norm(np.diff(result.agent_b_position_m, axis=0), axis=1) > 0
    )
    assert (
        result.metadata["controller"]["type"]
        == "bounded proportional-derivative force controller"
    )


def test_missing_optional_dependency_has_actionable_message(monkeypatch) -> None:
    monkeypatch.setattr(simulation_task, "bullet", None)
    with pytest.raises(RuntimeError, match="MSVC C\\+\\+ Build Tools"):
        run_collaborative_task()
