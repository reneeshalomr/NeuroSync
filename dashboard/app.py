"""Streamlit dashboard for explicitly synthetic and simulated NeuroSync outputs."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from neurosync.dashboard.data import (
    DashboardDataError,
    confusion_matrix_frame,
    load_dashboard_data,
    summarize_band_power,
    summarize_model_metrics,
)

st.set_page_config(page_title="NeuroSync Research Dashboard", layout="wide")
st.title("NeuroSync | Human–Robot Collaboration Research")
st.warning(
    "SYNTHETIC EEG + SIMULATED BEHAVIOR — task labels are scripted ground truth. "
    "These views do not represent human recordings, clinical evidence, or real-robot validation.",
    icon="⚠️",
)

default_output = (
    "outputs/full_run"
    if Path("outputs/full_run/dashboard_manifest.json").is_file()
    else "outputs/dashboard_example"
)
output_path = st.sidebar.text_input("Experiment output directory", value=default_output)
st.sidebar.caption("Choose a project output folder containing dashboard_manifest.json.")
try:
    data = load_dashboard_data(output_path)
except DashboardDataError as exc:
    st.error(str(exc))
    st.stop()

run_names = [run.run_id for run in data.runs]
selected_id = st.sidebar.selectbox("Simulation run", run_names)
run = next(item for item in data.runs if item.run_id == selected_id)
tabs = st.tabs(
    [
        "Overview",
        "Robot simulation",
        "EEG analysis",
        "Machine learning",
        "Adaptive assistance",
    ]
)

with tabs[0]:
    st.subheader("Experiment overview")
    left, right, third = st.columns(3)
    left.metric("Selected run", run.run_id)
    right.metric("Simulation seed", str(run.simulation_metadata.get("seed", "unknown")))
    third.metric(
        "Simulation engine", str(run.simulation_metadata.get("engine", "unknown"))
    )
    st.write(
        "**Condition:** scripted shared-workspace handover; "
        f"{len(run.events)} authored task phases"
    )
    st.write(
        "**Status:** generated artifacts loaded from the selected output directory"
    )
    st.write(
        "**Provenance:** synthetic EEG; PyBullet simulation; scripted task-state labels"
    )
    if data.model_report:
        st.dataframe(
            summarize_model_metrics(data.model_report),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Model report is not present in this dashboard bundle.")
    st.caption(
        "Correlation, coherence, and model metrics are descriptive for the constructed simulation protocol."
    )

with tabs[1]:
    st.subheader("PyBullet proxy simulation")
    sim = run.simulation
    positions = go.Figure()
    positions.add_trace(
        go.Scatter3d(
            x=sim["agent_a_position_m"][:, 0],
            y=sim["agent_a_position_m"][:, 1],
            z=sim["agent_a_position_m"][:, 2],
            mode="lines",
            name="Agent A proxy",
        )
    )
    positions.add_trace(
        go.Scatter3d(
            x=sim["agent_b_position_m"][:, 0],
            y=sim["agent_b_position_m"][:, 1],
            z=sim["agent_b_position_m"][:, 2],
            mode="lines",
            name="Robot proxy",
        )
    )
    positions.update_layout(title="Workspace trajectories (m)", scene_aspectmode="data")
    st.plotly_chart(positions, use_container_width=True)
    t = sim["time_seconds"]
    telemetry = go.Figure()
    telemetry.add_trace(
        go.Scatter(
            x=t, y=sim["inter_agent_distance_m"], name="Inter-agent distance (m)"
        )
    )
    telemetry.add_trace(
        go.Scatter(x=t, y=sim["agent_a_speed_m_s"], name="Agent A speed (m/s)")
    )
    telemetry.add_trace(
        go.Scatter(x=t, y=sim["agent_b_speed_m_s"], name="Robot proxy speed (m/s)")
    )
    for event in run.events:
        telemetry.add_vrect(
            x0=event["start_time_seconds"],
            x1=event["end_time_seconds"],
            fillcolor="orange" if event["task_state"] == "desynchronized" else "green",
            opacity=0.08,
            line_width=0,
            annotation_text=f"{event['phase_name']}: {event['task_state']}",
        )
    telemetry.update_layout(
        title="Distance, speed, and scripted task phases",
        xaxis_title="Simulation time (s)",
    )
    st.plotly_chart(telemetry, use_container_width=True)
    columns = [
        "phase_name",
        "task_state",
        "event_start_seconds",
        "event_end_seconds",
        "mean_inter_agent_distance_m",
        "agent_a_mean_speed_m_s",
        "agent_b_mean_speed_m_s",
    ]
    st.dataframe(run.features[columns], use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Synthetic EEG analysis")
    participant = st.selectbox("Participant", ["Participant A", "Participant B"])
    channel_names = [str(name) for name in run.eeg["channel_names"]]
    channel = st.selectbox("EEG channel", channel_names)
    participant_key = (
        "participant_a" if participant == "Participant A" else "participant_b"
    )
    channel_index = channel_names.index(channel)
    eeg_time = run.eeg["time_seconds"]
    signal_uv = run.eeg[participant_key][channel_index] * 1e6
    eeg_figure = go.Figure(
        go.Scattergl(
            x=eeg_time, y=signal_uv, mode="lines", name=f"{participant} {channel}"
        )
    )
    eeg_figure.update_layout(
        title="Synthetic EEG waveform",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (µV)",
    )
    st.plotly_chart(eeg_figure, use_container_width=True)
    bands = ("delta", "theta", "alpha", "beta", "gamma")
    prefix = "eeg_a_" if participant == "Participant A" else "eeg_b_"
    power_columns = [f"{prefix}mean_{band}_power_v2" for band in bands]
    if set(power_columns).issubset(run.features.columns):
        power = summarize_band_power(
            run.features, "a" if participant == "Participant A" else "b"
        )
        power_figure = go.Figure(go.Bar(x=power["band"], y=power["absolute_power_v2"]))
        power_figure.update_layout(
            title="Event-level mean band power (V²)",
            xaxis_title="Band",
            yaxis_title="Power (V²)",
        )
        st.plotly_chart(power_figure, use_container_width=True)
    coherence_columns = [f"eeg_mean_{band}_coherence" for band in bands]
    if set(coherence_columns).issubset(run.features.columns):
        coherence_values = run.features[coherence_columns].mean(axis=0).to_numpy()
        figure = go.Figure(go.Bar(x=list(bands), y=coherence_values))
        figure.update_layout(
            title="Descriptive magnitude-squared coherence",
            yaxis_title="Coherence (unitless)",
            xaxis_title="Band",
        )
        st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "Band powers are descriptive features. Coherence is a statistical association and does not prove interpersonal neural synchrony."
    )

with tabs[3]:
    st.subheader("Group-aware model comparison")
    if data.model_report:
        report = data.model_report
        st.info(
            f"Target source: {report.get('evaluation_metadata', {}).get('target_source', 'scripted simulation labels')}. Data and labels are synthetic/simulated; this does not estimate human-state prediction validity."
        )
        summary = summarize_model_metrics(report)
        figure = go.Figure(
            [
                go.Bar(
                    name="Balanced accuracy",
                    x=summary["model"],
                    y=summary["balanced_accuracy"],
                ),
                go.Bar(name="Macro F1", x=summary["model"], y=summary["macro_f1"]),
            ]
        )
        figure.update_layout(
            barmode="group", title="Out-of-fold model metrics", yaxis_range=[0, 1]
        )
        st.plotly_chart(figure, use_container_width=True)
        for model, metric in report["metrics"].items():
            with st.expander(f"{model} confusion matrix"):
                labels = metric["class_order"]
                st.write(f"Synthetic ground-truth classes: {', '.join(labels)}")
                st.dataframe(confusion_matrix_frame(metric), use_container_width=True)
    else:
        st.warning("No model report is available in this output directory.")

with tabs[4]:
    st.subheader("Adaptive assistance policy trace")
    if data.assistance_report:
        st.json(data.assistance_report.get("metrics", {}))
        st.caption(
            "Descriptive paired simulation metrics; no human benefit or real-robot safety claim."
        )
    decisions = run.assistance_decisions
    if decisions.empty:
        st.info(
            "The selected run has no assistance policy decisions. Select an assisted run or generate the example bundle."
        )
    elif {"time_seconds", "assistance_level", "reason"}.issubset(decisions.columns):
        figure = go.Figure(
            go.Scatter(
                x=decisions["time_seconds"],
                y=decisions["assistance_level"],
                mode="markers+lines",
                text=decisions["reason"],
            )
        )
        figure.update_layout(
            title="Logged policy decision over simulation time",
            xaxis_title="Time (s)",
            yaxis_title="Assistance level",
        )
        st.plotly_chart(figure, use_container_width=True)
        st.dataframe(decisions, use_container_width=True, hide_index=True)
        st.caption(
            "Policy inputs include scripted task state and proxy kinematics; actions change simulated controller settings only."
        )
    else:
        st.warning("Assistance decision table is missing required display columns.")
