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


class _FakeBullet:
    DIRECT = 0
    GEOM_BOX = 1
    GEOM_SPHERE = 2

    def __init__(self) -> None:
        self.next_id = 0

    def connect(self, mode):
        return 1

    def resetSimulation(self, **kwargs):
        pass

    def setGravity(self, *args, **kwargs):
        pass

    def setTimeStep(self, *args, **kwargs):
        pass

    def createCollisionShape(self, *args, **kwargs):
        self.next_id += 1
        return self.next_id

    def createMultiBody(self, *args, **kwargs):
        self.next_id += 1
        return self.next_id

    def resetBasePositionAndOrientation(self, *args, **kwargs):
        pass

    def stepSimulation(self, **kwargs):
        pass

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
    assert not np.array_equal(first.agent_b_position_m, synchronized.agent_b_position_m)


def test_npz_roundtrip_uses_pickle_free_metadata(fake_bullet, tmp_path) -> None:
    result = run_collaborative_task(CollaborativeTaskConfig(sampling_frequency_hz=10))
    target = save_simulation_result(result, tmp_path / "handover_simulated_run.npz")
    with np.load(target, allow_pickle=False) as archive:
        assert archive["agent_a_position_m"].shape == (61, 3)
        events = json.loads(str(archive["events_json"]))
        metadata = json.loads(str(archive["metadata_json"]))
        assert len(events) == len(result.events)
        assert metadata["data_source"] == "simulated"


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


def test_missing_optional_dependency_has_actionable_message(monkeypatch) -> None:
    monkeypatch.setattr(simulation_task, "bullet", None)
    with pytest.raises(RuntimeError, match="MSVC C\\+\\+ Build Tools"):
        run_collaborative_task()
