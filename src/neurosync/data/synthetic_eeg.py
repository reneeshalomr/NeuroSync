"""Generate reproducible, explicitly synthetic two-participant EEG-like signals.

The signal construction is a software fixture, not a biophysical head model or
an approximation validated against human EEG. Shared oscillations are injected
by construction in the high condition; this does not demonstrate interpersonal
neural synchrony, interaction, or any psychological state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
EEG_FREQUENCY_BANDS_HZ = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
INDEPENDENT_BAND_AMPLITUDES_UV = {
    "delta": 2.4,
    "theta": 1.8,
    "alpha": 2.2,
    "beta": 1.4,
    "gamma": 0.8,
}
SHARED_COMPONENT_BANDS = ((2.0, 4.0, 1.0), (8.0, 13.0, 0.8), (18.0, 28.0, 0.55))
AMPLITUDE_MODULATION_FREQUENCY_HZ = (0.05, 0.35)
SLOW_BACKGROUND_FREQUENCY_HZ = (0.2, 1.2)
MAX_BASE_OSCILLATORY_FREQUENCY_HZ = 45.0
MAX_OSCILLATORY_FREQUENCY_HZ = (
    MAX_BASE_OSCILLATORY_FREQUENCY_HZ + AMPLITUDE_MODULATION_FREQUENCY_HZ[1]
)


class ExperimentalCondition(str, Enum):
    """Synthetic signal-generation condition; not a human experimental result."""

    HIGH_SHARED_COMPONENT = "HIGH_SHARED_COMPONENT"
    LOW_SHARED_COMPONENT = "LOW_SHARED_COMPONENT"


@dataclass(frozen=True)
class SyntheticEEGData:
    """Arrays and provenance for one simulated two-participant recording.

    Signals have shape ``(channels, samples)`` and are expressed in volts (V).
    The time vector starts at zero and excludes the endpoint.
    """

    participant_a: FloatArray
    participant_b: FloatArray
    shared_component: FloatArray
    sampling_frequency_hz: float
    channel_names: tuple[str, ...]
    time_seconds: FloatArray
    condition: ExperimentalCondition
    random_seed: int
    generation_parameters: dict[str, Any]
    data_source: str = "synthetic"
    sample_task_state: NDArray[np.str_] | None = None
    task_events: tuple[dict[str, Any], ...] = ()
    simulation_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        """Number of samples in each participant's signal."""
        return int(self.time_seconds.size)

    @property
    def n_channels(self) -> int:
        """Number of simulated EEG channels per participant."""
        return len(self.channel_names)


def _validate_positive_number(name: str, value: float) -> float:
    if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


def _band_signal(
    rng: np.random.Generator,
    time_seconds: FloatArray,
    low_hz: float,
    high_hz: float,
    amplitude_uv: float,
) -> FloatArray:
    """Return a simple sinusoidal mixture within a named EEG frequency range."""
    frequency_hz = rng.uniform(low_hz, high_hz)
    phase_rad = rng.uniform(0.0, 2.0 * np.pi)
    modulation_hz = rng.uniform(*AMPLITUDE_MODULATION_FREQUENCY_HZ)
    modulation_phase = rng.uniform(0.0, 2.0 * np.pi)
    modulation = 0.8 + 0.2 * np.sin(
        2.0 * np.pi * modulation_hz * time_seconds + modulation_phase
    )
    return (
        amplitude_uv
        * modulation
        * np.sin(2.0 * np.pi * frequency_hz * time_seconds + phase_rad)
    )


def _participant_signal(
    rng: np.random.Generator,
    time_seconds: FloatArray,
    shared_components_uv: FloatArray,
    channel_count: int,
    noise_std_uv: float,
) -> FloatArray:
    """Construct channels from shared input, independent bands, and noise."""
    bands = tuple(
        (*EEG_FREQUENCY_BANDS_HZ[band], amplitude_uv)
        for band, amplitude_uv in INDEPENDENT_BAND_AMPLITUDES_UV.items()
    )
    samples = time_seconds.size
    signal_uv = np.empty((channel_count, samples), dtype=np.float64)
    for channel in range(channel_count):
        # Channel gain and phase vary; each participant also has an independent
        # mixture and independent broadband noise. Noise values are in microvolts.
        channel_gain = rng.uniform(0.65, 1.35)
        channel_signal = channel_gain * shared_components_uv
        for low_hz, high_hz, amplitude_uv in bands:
            channel_signal = channel_signal + _band_signal(
                rng, time_seconds, low_hz, high_hz, amplitude_uv * rng.uniform(0.6, 1.4)
            )
        white_noise = rng.normal(0.0, noise_std_uv, size=samples)
        slow_background = _band_signal(
            rng, time_seconds, *SLOW_BACKGROUND_FREQUENCY_HZ, 1.0
        )
        signal_uv[channel] = channel_signal + white_noise + slow_background
    # Convert microvolts to volts, the SI unit used by common EEG tooling.
    return signal_uv * 1e-6


def generate_synthetic_eeg(
    *,
    sampling_frequency_hz: float = 256.0,
    duration_seconds: float = 10.0,
    n_channels: int = 8,
    seed: int = 42,
    condition: ExperimentalCondition
    | str = ExperimentalCondition.HIGH_SHARED_COMPONENT,
    shared_component_amplitude_uv: float = 1.5,
    noise_std_uv: float = 2.0,
) -> SyntheticEEGData:
    """Generate a seeded two-participant EEG-like signal fixture.

    Args:
        sampling_frequency_hz: Samples per second. Must exceed 90.7 Hz to
            represent the upper gamma component including its modulation sideband.
        duration_seconds: Positive recording duration. Sample count is
            ``round(duration_seconds * sampling_frequency_hz)``.
        n_channels: Positive number of channels generated for each participant.
        seed: Non-negative integer seed for all random choices.
        condition: High or low injected shared-component setting. The low
            condition retains a small residual shared component (5% amplitude).
        shared_component_amplitude_uv: Base shared-component amplitude in µV.
        noise_std_uv: Standard deviation of independent white background noise
            in µV; an additional low-frequency background component is included.

    Returns:
        A ``SyntheticEEGData`` record with ``(channels, samples)`` arrays in V,
        a zero-based time vector, and explicit synthetic provenance.

    Raises:
        ValueError: If parameters are invalid or the sampling rate cannot
            represent the 45.35 Hz maximum oscillatory component.
    """
    sfreq = _validate_positive_number("sampling_frequency_hz", sampling_frequency_hz)
    duration = _validate_positive_number("duration_seconds", duration_seconds)
    shared_amplitude = _validate_positive_number(
        "shared_component_amplitude_uv", shared_component_amplitude_uv
    )
    noise_std = _validate_positive_number("noise_std_uv", noise_std_uv)
    if sfreq / 2.0 <= MAX_OSCILLATORY_FREQUENCY_HZ:
        raise ValueError(
            "sampling_frequency_hz must exceed 90.7 Hz for the 45.35 Hz maximum component"
        )
    if (
        isinstance(n_channels, bool)
        or not isinstance(n_channels, int)
        or n_channels < 1
    ):
        raise ValueError("n_channels must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    try:
        selected_condition = ExperimentalCondition(condition)
    except ValueError as exc:
        choices = ", ".join(item.value for item in ExperimentalCondition)
        raise ValueError(f"condition must be one of: {choices}") from exc

    sample_count = round(sfreq * duration)
    if sample_count < 2:
        raise ValueError(
            "duration_seconds and sampling_frequency_hz must produce at least 2 samples"
        )
    time_seconds = np.arange(sample_count, dtype=np.float64) / sfreq
    rng = np.random.Generator(np.random.PCG64(seed))

    shared_scale = (
        1.0
        if selected_condition is ExperimentalCondition.HIGH_SHARED_COMPONENT
        else 0.05
    )
    shared_components_uv = np.zeros(sample_count, dtype=np.float64)
    for low_hz, high_hz, relative_amplitude in SHARED_COMPONENT_BANDS:
        shared_components_uv += _band_signal(
            rng,
            time_seconds,
            low_hz,
            high_hz,
            shared_amplitude * relative_amplitude * shared_scale,
        )
    shared_component_volts = shared_components_uv * 1e-6

    channel_names = tuple(f"EEG{index + 1:03d}" for index in range(n_channels))
    participant_a = _participant_signal(
        rng, time_seconds, shared_components_uv, n_channels, noise_std
    )
    participant_b = _participant_signal(
        rng, time_seconds, shared_components_uv, n_channels, noise_std
    )
    parameters: dict[str, Any] = {
        "sampling_frequency_hz": sfreq,
        "duration_seconds_requested": duration,
        "sample_count": sample_count,
        "n_channels": n_channels,
        "seed": seed,
        "condition": selected_condition.value,
        "shared_component_amplitude_uv": shared_amplitude,
        "shared_component_scale": shared_scale,
        "noise_std_uv": noise_std,
        "maximum_oscillatory_frequency_hz": MAX_OSCILLATORY_FREQUENCY_HZ,
        "maximum_base_oscillatory_frequency_hz": MAX_BASE_OSCILLATORY_FREQUENCY_HZ,
        "nyquist_frequency_hz": sfreq / 2.0,
        "random_generator": "NumPy PCG64",
        "numpy_version": np.__version__,
        "amplitude_modulation_frequency_hz": list(AMPLITUDE_MODULATION_FREQUENCY_HZ),
        "slow_background_frequency_hz": list(SLOW_BACKGROUND_FREQUENCY_HZ),
        "slow_background_amplitude_uv": 1.0,
        "shared_component_bands": [list(band) for band in SHARED_COMPONENT_BANDS],
        "independent_band_amplitudes_uv": INDEPENDENT_BAND_AMPLITUDES_UV,
        "signal_unit": "V",
        "frequency_bands_hz": {
            band: list(frequency_range)
            for band, frequency_range in EEG_FREQUENCY_BANDS_HZ.items()
        },
    }
    return SyntheticEEGData(
        participant_a=participant_a,
        participant_b=participant_b,
        shared_component=shared_component_volts,
        sampling_frequency_hz=sfreq,
        channel_names=channel_names,
        time_seconds=time_seconds,
        condition=selected_condition,
        random_seed=seed,
        generation_parameters=parameters,
    )


def save_synthetic_eeg(data: SyntheticEEGData, path: str | Path) -> Path:
    """Save a synthetic recording as compressed NPZ plus JSON metadata.

    Arrays are loaded with ``allow_pickle=False``. The output filename must
    contain ``synthetic``. This function adds no implicit directory outside
    the path supplied by the caller.
    """
    output_path = Path(path)
    if "synthetic" not in output_path.stem.casefold():
        raise ValueError(
            "output filename must include 'synthetic' to preserve data provenance"
        )
    if output_path.suffix and output_path.suffix.casefold() != ".npz":
        raise ValueError("output path must have an .npz extension or no extension")
    if output_path.suffix.casefold() == ".npz" and output_path.suffix != ".npz":
        output_path = output_path.with_suffix(".npz")
    metadata = {
        "data_source": "synthetic",
        "scientific_status": "simulated signal fixture; not human EEG",
        "interpretation_limit": (
            "Injected shared components are generator-controlled statistical structure; "
            "they do not establish interpersonal neural synchrony or causal interaction."
        ),
        "sampling_frequency_hz": data.sampling_frequency_hz,
        "channel_names": list(data.channel_names),
        "condition": data.condition.value,
        "random_seed": data.random_seed,
        "generation_parameters": data.generation_parameters,
        "simulation_metadata": data.simulation_metadata,
        "task_events": list(data.task_events),
        "signal_unit": "V",
        "signal_shape": list(data.participant_a.shape),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        participant_a=data.participant_a,
        participant_b=data.participant_b,
        shared_component=data.shared_component,
        sampling_frequency_hz=np.asarray(data.sampling_frequency_hz),
        channel_names=np.asarray(data.channel_names, dtype=np.str_),
        time_seconds=data.time_seconds,
        condition=np.asarray(data.condition.value),
        random_seed=np.asarray(data.random_seed, dtype=np.int64),
        generation_parameters_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        sample_task_state=(
            data.sample_task_state
            if data.sample_task_state is not None
            else np.asarray([], dtype=np.str_)
        ),
        task_events_json=np.asarray(json.dumps(data.task_events, sort_keys=True)),
        simulation_metadata_json=np.asarray(
            json.dumps(data.simulation_metadata, sort_keys=True)
        ),
    )
    # np.savez appends .npz when callers omit it; return the actual filename.
    if not output_path.suffix:
        output_path = output_path.with_suffix(".npz")
    return output_path
