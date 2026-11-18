"""Manifest-driven loading and lightweight summaries for dashboard artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class DashboardDataError(ValueError):
    """Raised when dashboard artifacts are missing or violate their contract."""


@dataclass(frozen=True)
class DashboardRun:
    """One simulation, synthetic EEG record, and aligned feature table."""

    run_id: str
    simulation: dict[str, NDArray[Any]]
    simulation_metadata: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    eeg: dict[str, NDArray[Any]]
    eeg_metadata: dict[str, Any]
    features: pd.DataFrame
    assistance_decisions: pd.DataFrame


@dataclass(frozen=True)
class DashboardData:
    """Loaded dashboard view with optional model and assistance reports."""

    root: Path
    runs: tuple[DashboardRun, ...]
    model_report: dict[str, Any] | None
    assistance_report: dict[str, Any] | None
    manifest: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DashboardDataError(
            f"Cannot read JSON artifact {path.name}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise DashboardDataError(f"JSON artifact {path.name} must contain an object")
    return value


def _resolve_file(root: Path, name: Any, *, context: str) -> Path:
    if not isinstance(name, str) or not name:
        raise DashboardDataError(f"Manifest is missing {context} filename")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise DashboardDataError(f"Manifest {context} path must stay under output root")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise DashboardDataError(f"Manifest {context} path must stay under output root")
    if not path.is_file():
        raise DashboardDataError(f"Missing {context} artifact: {path}")
    return path


def _load_npz(path: Path, *, fields: tuple[str, ...]) -> dict[str, NDArray[Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = set(fields) - set(archive.files)
            if missing:
                raise DashboardDataError(
                    f"{path.name} is missing fields: {', '.join(sorted(missing))}"
                )
            return {name: np.asarray(archive[name]).copy() for name in fields}
    except (OSError, ValueError) as exc:
        if isinstance(exc, DashboardDataError):
            raise
        raise DashboardDataError(
            f"Cannot load NPZ artifact {path.name}: {exc}"
        ) from exc


def _validate_finite(label: str, arrays: dict[str, NDArray[Any]]) -> None:
    for name, values in arrays.items():
        if values.dtype.kind in "iufc" and not np.isfinite(values).all():
            raise DashboardDataError(f"{label} field {name} contains non-finite values")


def load_dashboard_data(output_dir: str | Path) -> DashboardData:
    """Load one dashboard bundle described by ``dashboard_manifest.json``.

    NPZ inputs are opened with ``allow_pickle=False``. All paths are relative to
    ``output_dir`` and may not escape it. Synthetic EEG and simulated behavior
    provenance are mandatory for every run.
    """
    root = Path(output_dir).expanduser().resolve()
    manifest_path = root / "dashboard_manifest.json"
    if not manifest_path.is_file():
        raise DashboardDataError(
            f"Dashboard manifest not found at {manifest_path}. Generate an example with "
            "`.venv\\Scripts\\python.exe examples\\build_dashboard_example.py`."
        )
    manifest = _read_json(manifest_path)
    run_specs = manifest.get("runs")
    if not isinstance(run_specs, list) or not run_specs:
        raise DashboardDataError("Dashboard manifest must define at least one run")
    runs: list[DashboardRun] = []
    for index, spec in enumerate(run_specs):
        if not isinstance(spec, dict):
            raise DashboardDataError(f"Manifest run {index} must be an object")
        sim_path = _resolve_file(
            root, spec.get("simulation_file"), context="simulation"
        )
        sim_fields = (
            "time_seconds",
            "agent_a_position_m",
            "agent_b_position_m",
            "agent_a_velocity_m_s",
            "agent_b_velocity_m_s",
            "agent_a_speed_m_s",
            "agent_b_speed_m_s",
            "inter_agent_distance_m",
            "events_json",
            "metadata_json",
            "assistance_decisions_json",
        )
        sim = _load_npz(sim_path, fields=sim_fields)
        _validate_finite("simulation", sim)
        sim_metadata = json.loads(str(sim["metadata_json"].item()))
        events = tuple(json.loads(str(sim["events_json"].item())))
        decisions = pd.DataFrame(
            json.loads(str(sim["assistance_decisions_json"].item()))
        )
        if (
            sim_metadata.get("data_source") != "simulated"
            or sim_metadata.get("engine") != "PyBullet DIRECT"
        ):
            raise DashboardDataError(
                "Simulation artifact must identify real PyBullet simulation provenance"
            )

        eeg_path = _resolve_file(root, spec.get("eeg_file"), context="synthetic EEG")
        eeg_fields = (
            "time_seconds",
            "participant_a",
            "participant_b",
            "channel_names",
            "metadata_json",
        )
        eeg = _load_npz(eeg_path, fields=eeg_fields)
        _validate_finite("EEG", eeg)
        eeg_metadata = json.loads(str(eeg["metadata_json"].item()))
        if (
            eeg_metadata.get("data_source") != "synthetic"
            or eeg_metadata.get("signal_unit") != "V"
        ):
            raise DashboardDataError(
                "EEG artifact must be explicitly synthetic and in volts"
            )
        names = tuple(str(name) for name in eeg["channel_names"].tolist())
        eeg["channel_names"] = np.asarray(names)
        expected = (len(names), eeg["time_seconds"].size)
        if (
            eeg["participant_a"].shape != expected
            or eeg["participant_b"].shape != expected
        ):
            raise DashboardDataError(
                "EEG array dimensions do not match time and channel names"
            )

        feature_path = _resolve_file(
            root, spec.get("features_file"), context="feature table"
        )
        features = pd.read_csv(feature_path)
        required_columns = {
            "task_state",
            "phase_name",
            "event_start_seconds",
            "event_end_seconds",
        }
        if not required_columns.issubset(features.columns):
            raise DashboardDataError(
                "Feature table does not contain the dashboard event columns"
            )
        if not np.isfinite(
            features.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        ).all():
            raise DashboardDataError("Feature table contains non-finite numeric values")
        decision_file = spec.get("assistance_file")
        if decision_file:
            decisions_path = _resolve_file(
                root, decision_file, context="assistance decisions"
            )
            decisions = pd.read_csv(decisions_path)
        run_id = str(spec.get("run_id", f"run-{index + 1}"))
        runs.append(
            DashboardRun(
                run_id,
                sim,
                sim_metadata,
                events,
                eeg,
                eeg_metadata,
                features,
                decisions,
            )
        )

    report_name = manifest.get("model_report_file")
    assistance_name = manifest.get("assistance_report_file")
    model_report = (
        _read_json(_resolve_file(root, report_name, context="model report"))
        if report_name
        else None
    )
    assistance_report = (
        _read_json(_resolve_file(root, assistance_name, context="assistance report"))
        if assistance_name
        else None
    )
    return DashboardData(root, tuple(runs), model_report, assistance_report, manifest)


def summarize_model_metrics(report: dict[str, Any]) -> pd.DataFrame:
    """Return stable rows for model comparison metrics, validating values."""
    metrics = report.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise DashboardDataError("Model report must contain a non-empty metrics object")
    rows: list[dict[str, float | str]] = []
    for model, values in metrics.items():
        if not isinstance(values, dict):
            raise DashboardDataError(f"Metrics for {model!r} must be an object")
        row: dict[str, float | str] = {"model": str(model)}
        for key in ("balanced_accuracy", "macro_f1"):
            value = values.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise DashboardDataError(f"Model {model!r} has invalid {key}")
            row[key] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_band_power(features: pd.DataFrame, participant: str) -> pd.DataFrame:
    """Average event-level absolute power across the five configured EEG bands."""
    if participant not in {"a", "b"}:
        raise DashboardDataError("participant must be 'a' or 'b'")
    bands = ("delta", "theta", "alpha", "beta", "gamma")
    columns = [f"eeg_{participant}_mean_{band}_power_v2" for band in bands]
    if not set(columns).issubset(features.columns):
        raise DashboardDataError(
            "Feature table is missing participant band-power summaries"
        )
    values = features[columns].to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise DashboardDataError("Band-power features must be non-empty and finite")
    return pd.DataFrame({"band": bands, "absolute_power_v2": values.mean(axis=0)})


def confusion_matrix_frame(metric: dict[str, Any]) -> pd.DataFrame:
    """Format and validate one report confusion matrix for display."""
    labels = metric.get("class_order")
    matrix = np.asarray(metric.get("confusion_matrix"))
    if (
        not isinstance(labels, list)
        or not labels
        or matrix.shape != (len(labels), len(labels))
        or matrix.dtype.kind not in "iuf"
        or not np.isfinite(matrix).all()
    ):
        raise DashboardDataError("Confusion matrix dimensions or values are invalid")
    return pd.DataFrame(matrix, index=labels, columns=labels).rename_axis("true class")
