"""Run a headless, scripted PyBullet handover and save simulated telemetry."""

from __future__ import annotations

from neurosync.simulation import (
    CollaborativeTaskConfig,
    run_collaborative_task,
    save_simulation_result,
)


def main() -> None:
    config = CollaborativeTaskConfig(seed=2026)
    result = run_collaborative_task(config)
    path = save_simulation_result(
        result,
        "data/generated/handover_simulated_seed2026.npz",
    )
    print(
        "Simulation only; the kinematic proxies are not human recordings or a validated robot."
    )
    print(f"Saved: {path}; samples: {result.time_seconds.size}")
    print(f"Scripted task-state events: {len(result.events)}")
    print(f"Whole-run speed Pearson correlation: {result.velocity_correlation:.3f}")
    print(f"Idle time (s): {result.idle_duration_seconds}")


if __name__ == "__main__":
    main()
