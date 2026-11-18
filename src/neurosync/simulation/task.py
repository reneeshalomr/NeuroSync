"""Headless PyBullet scripted handover task and synthetic telemetry.

Two dynamic proxy spheres track authored trajectories under a bounded PD
controller while a PyBullet DIRECT world advances at a fixed step. Events are
ground truth from this schedule, not observations or annotations of behavior.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import scipy
from numpy.typing import NDArray

try:
    import pybullet as bullet
except ImportError:  # pragma: no cover - exercised in installs without [sim]
    bullet = None


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CollaborativeTaskConfig:
    """Fixed-step handover schedule, with explicit synchronized states."""

    sampling_frequency_hz: float = 20.0
    seed: int = 2026
    idle_speed_threshold_m_s: float = 0.015
    desynchronization_delay_fraction: float = 0.55
    phases: tuple[tuple[str, str, float], ...] = (
        ("approach", "synchronized", 2.0),
        ("handover", "desynchronized", 2.0),
        ("retreat", "synchronized", 2.0),
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.sampling_frequency_hz, bool)
            or not math.isfinite(self.sampling_frequency_hz)
            or self.sampling_frequency_hz <= 0
        ):
            raise ValueError("sampling_frequency_hz must be finite and positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if (
            not math.isfinite(self.idle_speed_threshold_m_s)
            or self.idle_speed_threshold_m_s < 0
        ):
            raise ValueError("idle_speed_threshold_m_s must be finite and non-negative")
        if (
            not math.isfinite(self.desynchronization_delay_fraction)
            or not 0 <= self.desynchronization_delay_fraction < 1
        ):
            raise ValueError("desynchronization_delay_fraction must be in [0, 1)")
        if not self.phases:
            raise ValueError("at least one task phase is required")
        allowed_states = {"synchronized", "desynchronized"}
        for name, state, duration in self.phases:
            if not name or state not in allowed_states:
                raise ValueError("phases require a name and a supported task state")
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("phase durations must be finite and positive")


@dataclass(frozen=True)
class TaskEvent:
    """Scripted task-state interval with a half-open [start, end) convention."""

    event_id: int
    phase_name: str
    task_state: str
    start_time_seconds: float
    end_time_seconds: float
    source: str = "scripted_simulation_ground_truth"


@dataclass(frozen=True)
class CollaborativeTaskResult:
    """Behavioral telemetry and provenance for one simulated task run."""

    time_seconds: FloatArray
    agent_a_position_m: FloatArray
    agent_b_position_m: FloatArray
    agent_a_velocity_m_s: FloatArray
    agent_b_velocity_m_s: FloatArray
    agent_a_speed_m_s: FloatArray
    agent_b_speed_m_s: FloatArray
    inter_agent_distance_m: FloatArray
    events: tuple[TaskEvent, ...]
    velocity_correlation: float
    idle_duration_seconds: dict[str, float]
    metadata: dict[str, Any]
    assistance_decisions: tuple[dict[str, Any], ...] = ()


class AssistancePolicy(Protocol):
    """Minimal controller policy contract consumed by the simulation loop."""

    policy_name: str

    def decide(self, observation: dict[str, Any]) -> dict[str, Any]: ...


def _trajectory(
    phase: str,
    progress_a: FloatArray,
    progress_b: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Return authored 3D kinematic paths for the two proxy agents."""
    homes = np.array([[-0.65, 0.0, 0.72], [0.65, 0.0, 0.72]])
    meeting_a = np.array([-0.09, 0.0, 0.88])
    meeting_b = np.array([0.09, 0.0, 0.88])
    if phase == "approach":
        a_start, a_end, b_start, b_end = homes[0], meeting_a, homes[1], meeting_b
    elif phase == "handover":
        a_start, a_end, b_start, b_end = meeting_a, meeting_a, meeting_b, meeting_a
    elif phase == "retreat":
        a_start, a_end, b_start, b_end = meeting_a, homes[0], meeting_a, homes[1]
    else:
        # Unknown phase names retain a safe stationary trajectory at the workspace center.
        center = np.array([0.0, 0.0, 0.8])
        a_start = a_end = b_start = b_end = center
    a = a_start + progress_a[:, None] * (a_end - a_start)
    b = b_start + progress_b[:, None] * (b_end - b_start)
    return a, b


def _idle_duration(speed: FloatArray, threshold: float, dt: float) -> float:
    """Sum sampled intervals below the explicit idle speed threshold."""
    return float(np.count_nonzero(speed < threshold) * dt)


def run_collaborative_task(
    config: CollaborativeTaskConfig = CollaborativeTaskConfig(),
    *,
    assistance_policy: AssistancePolicy | None = None,
) -> CollaborativeTaskResult:
    """Run a headless PyBullet shared-workspace handover with scripted agents.

    The proxies are dynamic spheres moved by a scripted PD controller.
    Reference trajectories are authored, not produced by a learned controller,
    human motion capture, or biomechanical model. Sampled positions and
    velocities are read from PyBullet after each physics step.
    Simulation state names are scheduled ground truth for software evaluation.
    """
    dt = 1.0 / config.sampling_frequency_hz
    duration = float(sum(phase[2] for phase in config.phases))
    n_steps = round(duration * config.sampling_frequency_hz)
    if n_steps < 2 or not math.isclose(
        n_steps / config.sampling_frequency_hz, duration, abs_tol=1e-9
    ):
        raise ValueError(
            "total phase duration must align to the configured fixed time step"
        )
    if bullet is None:
        raise RuntimeError(
            "PyBullet is required for the simulation; install NeuroSync with `.[sim]`. "
            "On Windows, PyBullet 3.2.7 currently builds from source and requires MSVC C++ Build Tools."
        )
    times = np.arange(n_steps + 1, dtype=np.float64) * dt
    positions_a = np.empty((times.size, 3), dtype=np.float64)
    positions_b = np.empty((times.size, 3), dtype=np.float64)
    events: list[TaskEvent] = []
    phase_ends: list[float] = []
    elapsed = 0.0
    for event_id, (phase_name, task_state, phase_duration) in enumerate(config.phases):
        end = elapsed + phase_duration
        events.append(TaskEvent(event_id, phase_name, task_state, elapsed, end))
        phase_ends.append(end)
        elapsed = end

    client_id = bullet.connect(bullet.DIRECT)
    if client_id < 0:
        raise RuntimeError("could not create a PyBullet DIRECT client")
    try:
        bullet.resetSimulation(physicsClientId=client_id)
        bullet.setGravity(0.0, 0.0, -9.81, physicsClientId=client_id)
        bullet.setTimeStep(dt, physicsClientId=client_id)
        plane_shape = bullet.createCollisionShape(
            bullet.GEOM_BOX, halfExtents=[1.0, 0.6, 0.025], physicsClientId=client_id
        )
        bullet.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=plane_shape,
            basePosition=[0.0, 0.0, 0.55],
            physicsClientId=client_id,
        )
        sphere_shape_a = bullet.createCollisionShape(
            bullet.GEOM_SPHERE, radius=0.055, physicsClientId=client_id
        )
        sphere_shape_b = bullet.createCollisionShape(
            bullet.GEOM_SPHERE, radius=0.055, physicsClientId=client_id
        )
        body_a = bullet.createMultiBody(
            baseMass=0.25,
            baseCollisionShapeIndex=sphere_shape_a,
            basePosition=[-0.65, 0.0, 0.72],
            physicsClientId=client_id,
        )
        body_b = bullet.createMultiBody(
            baseMass=0.25,
            baseCollisionShapeIndex=sphere_shape_b,
            basePosition=[0.65, 0.0, 0.72],
            physicsClientId=client_id,
        )
        velocity_a = np.zeros_like(positions_a)
        velocity_b = np.zeros_like(positions_b)
        positions_a[0] = [-0.65, 0.0, 0.72]
        positions_b[0] = [0.65, 0.0, 0.72]
        kp, kd, max_force_n = 36.0, 12.0, 8.0
        human_idle_duration = 0.0
        assistance_decisions: list[dict[str, Any]] = []
        for sample_index, time_s in enumerate(times[1:], start=1):
            phase_index = min(
                int(np.searchsorted(phase_ends, time_s, side="right")),
                len(config.phases) - 1,
            )
            phase_name, state, phase_duration = config.phases[phase_index]
            event = events[phase_index]
            local_progress = float(
                np.clip((time_s - event.start_time_seconds) / phase_duration, 0.0, 1.0)
            )
            progress_a = np.array([local_progress])
            current_a, _ = bullet.getBasePositionAndOrientation(
                body_a, physicsClientId=client_id
            )
            current_b, _ = bullet.getBasePositionAndOrientation(
                body_b, physicsClientId=client_id
            )
            velocity_a_current, _ = bullet.getBaseVelocity(
                body_a, physicsClientId=client_id
            )
            velocity_b_current, _ = bullet.getBaseVelocity(
                body_b, physicsClientId=client_id
            )
            human_speed = float(np.linalg.norm(velocity_a_current))
            robot_speed = float(np.linalg.norm(velocity_b_current))
            human_idle_duration = (
                human_idle_duration + dt
                if human_speed < config.idle_speed_threshold_m_s
                else 0.0
            )
            distance_now = float(
                np.linalg.norm(np.asarray(current_a) - np.asarray(current_b))
            )
            delay = config.desynchronization_delay_fraction
            robot_kp, robot_kd, robot_max_force = kp, kd, max_force_n
            if assistance_policy is not None:
                observation = {
                    "time_seconds": float(time_s),
                    "task_state": state,
                    "human_speed_m_s": human_speed,
                    "robot_speed_m_s": robot_speed,
                    "inter_agent_distance_m": distance_now,
                    "human_idle_duration_seconds": human_idle_duration,
                }
                action = assistance_policy.decide(observation)
                try:
                    delay = float(action["robot_target_delay_fraction"])
                    robot_kp = float(action["robot_kp_per_s2"])
                    robot_kd = float(action["robot_kd_per_s"])
                    robot_max_force = float(action["robot_max_force_n"])
                    assistance_level = str(action["assistance_level"])
                    reason = str(action["reason"])
                    policy_name = str(action["policy_name"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "assistance policy returned an invalid action"
                    ) from exc
                if (
                    not all(
                        math.isfinite(value)
                        for value in (delay, robot_kp, robot_kd, robot_max_force)
                    )
                    or not 0 <= delay < 1
                    or robot_kp < 0
                    or robot_kd < 0
                    or robot_max_force <= 0
                ):
                    raise ValueError("assistance action controller values are invalid")
                assistance_decisions.append(
                    {
                        **observation,
                        "policy_name": policy_name,
                        "assistance_level": assistance_level,
                        "reason": reason,
                        "robot_target_delay_fraction": delay,
                        "robot_kp_per_s2": robot_kp,
                        "robot_kd_per_s": robot_kd,
                        "robot_max_force_n": robot_max_force,
                    }
                )
            if state == "desynchronized":
                progress_b = np.array(
                    [np.clip((local_progress - delay) / (1.0 - delay), 0.0, 1.0)]
                )
            else:
                progress_b = np.array([local_progress])
            pos_a, pos_b = _trajectory(phase_name, progress_a, progress_b)
            pos_a, pos_b = pos_a[0], pos_b[0]
            controller_settings = (
                (body_a, pos_a, kp, kd, max_force_n),
                (body_b, pos_b, robot_kp, robot_kd, robot_max_force),
            )
            for body, target, body_kp, body_kd, body_max_force in controller_settings:
                current, _ = bullet.getBasePositionAndOrientation(
                    body, physicsClientId=client_id
                )
                velocity, _ = bullet.getBaseVelocity(body, physicsClientId=client_id)
                force = 0.25 * (
                    body_kp * (target - np.asarray(current))
                    - body_kd * np.asarray(velocity)
                )
                norm = float(np.linalg.norm(force))
                if norm > body_max_force:
                    force *= body_max_force / norm
                bullet.applyExternalForce(
                    body,
                    -1,
                    force.tolist(),
                    current,
                    bullet.WORLD_FRAME,
                    physicsClientId=client_id,
                )
            bullet.stepSimulation(physicsClientId=client_id)
            for body, positions, velocities in (
                (body_a, positions_a, velocity_a),
                (body_b, positions_b, velocity_b),
            ):
                position, _ = bullet.getBasePositionAndOrientation(
                    body, physicsClientId=client_id
                )
                velocity, _ = bullet.getBaseVelocity(body, physicsClientId=client_id)
                positions[sample_index] = position
                velocities[sample_index] = velocity
    finally:
        bullet.disconnect(physicsClientId=client_id)

    speed_a = np.linalg.norm(velocity_a, axis=1)
    speed_b = np.linalg.norm(velocity_b, axis=1)
    distance = np.linalg.norm(positions_a - positions_b, axis=1)
    if np.std(speed_a) == 0 or np.std(speed_b) == 0:
        velocity_corr = 0.0
    else:
        velocity_corr = float(np.corrcoef(speed_a, speed_b)[0, 1])
    if not np.isfinite(velocity_corr):
        velocity_corr = 0.0
    return CollaborativeTaskResult(
        time_seconds=times,
        agent_a_position_m=positions_a,
        agent_b_position_m=positions_b,
        agent_a_velocity_m_s=velocity_a,
        agent_b_velocity_m_s=velocity_b,
        agent_a_speed_m_s=speed_a,
        agent_b_speed_m_s=speed_b,
        inter_agent_distance_m=distance,
        events=tuple(events),
        velocity_correlation=velocity_corr,
        idle_duration_seconds={
            "agent_a": _idle_duration(
                speed_a[:-1], config.idle_speed_threshold_m_s, dt
            ),
            "agent_b": _idle_duration(
                speed_b[:-1], config.idle_speed_threshold_m_s, dt
            ),
        },
        metadata={
            "data_source": "simulated",
            "scientific_status": "scripted PyBullet dynamic proxy task with PD trajectory tracking; not human behavior or a validated robot system",
            "task_name": "shared_workspace_handover",
            "engine": "PyBullet DIRECT",
            "pybullet_version": getattr(bullet, "__version__", "3.2.7"),
            "scipy_version": scipy.__version__,
            "seed": config.seed,
            "seed_role": "run identifier; authored trajectories and controller are deterministic",
            "controller": {
                "type": "bounded proportional-derivative force controller",
                "kp_per_s2": kp,
                "kd_per_s": kd,
                "maximum_force_n": max_force_n,
                "body_mass_kg": 0.25,
            },
            "assistance_policy": {
                "enabled": assistance_policy is not None,
                "name": getattr(assistance_policy, "policy_name", None),
                "decision_count": len(assistance_decisions),
                "decision_source": "per-step simulated kinematics and scripted task phase",
            },
            "sampling_frequency_hz": config.sampling_frequency_hz,
            "time_step_seconds": dt,
            "duration_seconds": duration,
            "phases": [asdict(event) for event in events],
            "task_state_ground_truth_source": "scripted phase schedule",
            "agent_labels": ["simulated_human_proxy", "simulated_robot_proxy"],
            "position_unit": "m",
            "velocity_unit": "m/s",
            "idle_speed_threshold_m_s": config.idle_speed_threshold_m_s,
            "desynchronization_delay_fraction": config.desynchronization_delay_fraction,
            "velocity_correlation_definition": "Pearson correlation of per-step speed magnitudes over the full run",
            "limitations": [
                "Reference trajectories are scripted; dynamic proxy bodies do not represent human or robot biomechanics.",
                "Task states are authored ground truth, not inferred collaboration quality.",
                "This environment is a software fixture, not a safety or deployment evaluation.",
            ],
        },
        assistance_decisions=tuple(assistance_decisions),
    )


def save_simulation_result(result: CollaborativeTaskResult, path: str | Path) -> Path:
    """Save numeric telemetry and JSON provenance in a pickle-free NPZ file."""
    output = Path(path)
    if "simulated" not in output.stem.casefold():
        raise ValueError("simulation output filename must include 'simulated'")
    if output.suffix.lower() != ".npz":
        raise ValueError("simulation result path must have an .npz extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    event_json = json.dumps([asdict(event) for event in result.events], sort_keys=True)
    assistance_json = json.dumps(result.assistance_decisions, sort_keys=True)
    np.savez_compressed(
        output,
        time_seconds=result.time_seconds,
        agent_a_position_m=result.agent_a_position_m,
        agent_b_position_m=result.agent_b_position_m,
        agent_a_velocity_m_s=result.agent_a_velocity_m_s,
        agent_b_velocity_m_s=result.agent_b_velocity_m_s,
        agent_a_speed_m_s=result.agent_a_speed_m_s,
        agent_b_speed_m_s=result.agent_b_speed_m_s,
        inter_agent_distance_m=result.inter_agent_distance_m,
        events_json=np.asarray(event_json),
        assistance_decisions_json=np.asarray(assistance_json),
        velocity_correlation=np.asarray(result.velocity_correlation),
        idle_duration_json=np.asarray(
            json.dumps(result.idle_duration_seconds, sort_keys=True)
        ),
        metadata_json=np.asarray(json.dumps(result.metadata, sort_keys=True)),
    )
    return output
