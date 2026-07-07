"""
===============================================================================
evaluation_plots.py
===============================================================================
Write minimal visible trajectory comparison outputs for MVP evaluation.

Responsibilities:
  - Validate comparable reference and actual sampled trajectories
  - Write headless matplotlib reference-vs-actual trajectory plots when available
  - Write notebook-ready trained-policy trace plots for final reports
  - Fall back to compact JSON comparison data when plotting is unavailable

Design principles:
  - Keep plotting independent from training, rollout collection, and notebooks
  - Ensure output paths are explicit and parent directories are created on demand

Boundaries:
  - Metric calculations belong in evaluation_trajectory_metrics.py
  - Rollout generation belongs in evaluation_rollout.py
===============================================================================

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src import evaluation

TRACE_POSITION_NDIM = 2
XYZ_DIMENSIONS = 3
CANONICAL_POLICY_PLOT_FILENAMES = {
    "trajectory_xy": "trajectory_xy.png",
    "trajectory_xyz": "trajectory_xyz.png",
    "position_error": "position_error.png",
    "action_trace": "action_trace.png",
}

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src import trajectories


@dataclass(frozen=True)
class RolloutTracePlotResult:
    """
    Summary returned after writing trained-policy trace plots.

    Parameters
    ----------
    plot_paths
        Mapping from stable plot names to generated PNG paths.
    output_kind
        Plot backend used for generated figures.
    step_count
        Number of rollout steps plotted.

    """

    plot_paths: dict[str, str]
    output_kind: str
    step_count: int


@dataclass(frozen=True)
class TrajectoryPlotResult:
    """
    Summary returned by trajectory comparison output writers.

    Parameters
    ----------
    output_path
        Path to the generated plot or fallback data file.
    output_kind
        Either ``"matplotlib_png"`` or ``"json_fallback"``.
    sample_count
        Number of trajectory samples included in the output.

    """

    output_path: str
    output_kind: str
    sample_count: int


def write_trajectory_comparison(
    reference: trajectories.primitives.Trajectory,
    actual: trajectories.primitives.Trajectory,
    output_path: str | Path,
    force_fallback: bool = False,
) -> TrajectoryPlotResult:
    """
    Write a reference-vs-actual trajectory comparison output.

    Parameters
    ----------
    reference
        Sampled reference trajectory.
    actual
        Sampled actual trajectory with matching time samples.
    output_path
        Preferred output path. A JSON fallback may replace the suffix with ``.json``.
    force_fallback
        Whether to skip matplotlib and write fallback JSON data directly.

    Returns
    -------
    TrajectoryPlotResult
        Output path, output kind, and sample count.

    """
    errors = evaluation.trajectory_metrics.compute_position_errors(reference=reference, actual=actual)
    resolved_path = Path(output_path)
    if force_fallback:
        return _write_json_fallback(reference=reference, actual=actual, errors=errors, output_path=resolved_path)
    try:
        return _write_matplotlib_plot(reference=reference, actual=actual, errors=errors, output_path=resolved_path)
    except ImportError:
        return _write_json_fallback(reference=reference, actual=actual, errors=errors, output_path=resolved_path.with_suffix(".json"))


def write_policy_rollout_trace_plots(
    trace_records_or_path: Sequence[Mapping[str, Any]] | str | Path,
    output_dir: str | Path,
) -> RolloutTracePlotResult:
    """
    Write notebook-ready trained-policy rollout plots from a saved trace.

    Parameters
    ----------
    trace_records_or_path
        Trace records or a JSONL path created by ``evaluation.rollout.write_policy_rollout_trace``.
    output_dir
        Directory where plot PNGs are written.

    Returns
    -------
    RolloutTracePlotResult
        Generated plot paths keyed by stable notebook-friendly names.

    Raises
    ------
    ValueError
        If the trace is empty or lacks required position/time fields.

    """
    records = _load_trace_records(trace_records_or_path)
    arrays = _trace_arrays(records)
    segments = _episode_segments(arrays)
    resolved_dir = Path(output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plot_paths: dict[str, str] = {}

    xy_path = resolved_dir / CANONICAL_POLICY_PLOT_FILENAMES["trajectory_xy"]
    figure, axis = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    for segment_index, segment in enumerate(segments):
        axis.plot(
            arrays["reference"][segment, 0],
            arrays["reference"][segment, 1],
            color="#1f77b4",
            linewidth=2.0,
            label="reference" if segment_index == 0 else None,
        )
        axis.plot(
            arrays["actual"][segment, 0],
            arrays["actual"][segment, 1],
            color="#d62728",
            linewidth=2.0,
            label="actual" if segment_index == 0 else None,
        )
    axis.set_xlabel("x m")
    axis.set_ylabel("y m")
    axis.set_title("XY reference vs actual")
    axis.axis("equal")
    axis.legend()
    figure.savefig(xy_path, dpi=140)
    plt.close(figure)
    plot_paths["trajectory_xy"] = str(xy_path)

    xyz_path = resolved_dir / CANONICAL_POLICY_PLOT_FILENAMES["trajectory_xyz"]
    figure, axes = plt.subplots(3, 1, figsize=(7.0, 6.0), sharex=True, constrained_layout=True)
    labels = ("x", "y", "z")
    for axis_index, axis in enumerate(axes):
        for segment_index, segment in enumerate(segments):
            axis.plot(
                arrays["time"][segment],
                arrays["reference"][segment, axis_index],
                color="#1f77b4",
                linewidth=1.8,
                label="reference" if axis_index == 0 and segment_index == 0 else None,
            )
            axis.plot(
                arrays["time"][segment],
                arrays["actual"][segment, axis_index],
                color="#d62728",
                linewidth=1.6,
                label="actual" if axis_index == 0 and segment_index == 0 else None,
            )
        _mark_tracking_starts(axis, arrays=arrays, segments=segments, label=axis_index == 0)
        axis.set_ylabel(f"{labels[axis_index]} m")
    axes[0].set_title("XYZ vs time")
    axes[-1].set_xlabel("time s")
    axes[0].legend()
    figure.savefig(xyz_path, dpi=140)
    plt.close(figure)
    plot_paths["trajectory_xyz"] = str(xyz_path)

    error_path = resolved_dir / CANONICAL_POLICY_PLOT_FILENAMES["position_error"]
    figure, axis = plt.subplots(figsize=(7.0, 3.5), constrained_layout=True)
    for segment in segments:
        axis.plot(arrays["time"][segment], arrays["position_error"][segment], color="#d62728", linewidth=2.0)
    _mark_tracking_starts(axis, arrays=arrays, segments=segments, label=True)
    axis.set_xlabel("time s")
    axis.set_ylabel("position error m")
    axis.set_title("Position error vs time")
    figure.savefig(error_path, dpi=140)
    plt.close(figure)
    plot_paths["position_error"] = str(error_path)

    action_array = _optional_action_array(records, key="action")
    if action_array is not None:
        action_path = resolved_dir / CANONICAL_POLICY_PLOT_FILENAMES["action_trace"]
        figure, axis = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
        for action_index in range(action_array.shape[1]):
            for segment_index, segment in enumerate(segments):
                axis.plot(
                    arrays["time"][segment],
                    action_array[segment, action_index],
                    linewidth=1.5,
                    label=f"action {action_index}" if segment_index == 0 else None,
                )
        _mark_tracking_starts(axis, arrays=arrays, segments=segments, label=True)
        axis.set_xlabel("time s")
        axis.set_ylabel("action")
        axis.set_title("Action vs time")
        axis.legend()
        figure.savefig(action_path, dpi=140)
        plt.close(figure)
        plot_paths["action_trace"] = str(action_path)

    return RolloutTracePlotResult(plot_paths=plot_paths, output_kind="matplotlib_png", step_count=len(records))


def _write_matplotlib_plot(
    reference: trajectories.primitives.Trajectory,
    actual: trajectories.primitives.Trajectory,
    errors: np.ndarray,
    output_path: Path,
) -> TrajectoryPlotResult:
    """Write a headless matplotlib trajectory comparison plot."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: PLC0415

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), constrained_layout=True)
    axes[0].plot(reference.positions[:, 0], reference.positions[:, 1], label="reference", linewidth=2.0)
    axes[0].plot(actual.positions[:, 0], actual.positions[:, 1], label="actual", linewidth=2.0, linestyle="--")
    axes[0].set_xlabel("x m")
    axes[0].set_ylabel("y m")
    axes[0].set_title("XY path")
    axes[0].axis("equal")
    axes[0].legend()

    axes[1].plot(reference.times, reference.positions[:, 2], label="reference z", linewidth=2.0)
    axes[1].plot(actual.times, actual.positions[:, 2], label="actual z", linewidth=2.0, linestyle="--")
    axes[1].plot(reference.times, errors, label="position error", linewidth=1.5)
    axes[1].set_xlabel("time s")
    axes[1].set_ylabel("meters")
    axes[1].set_title("Height and error")
    axes[1].legend()

    figure.savefig(output_path, dpi=120)
    plt.close(figure)
    return TrajectoryPlotResult(output_path=str(output_path), output_kind="matplotlib_png", sample_count=int(errors.shape[0]))


def _load_trace_records(trace_records_or_path: Sequence[Mapping[str, Any]] | str | Path) -> list[dict[str, Any]]:
    """Return trace records from an in-memory sequence or JSONL path."""
    if isinstance(trace_records_or_path, (str, Path)):
        return evaluation.rollout.load_policy_rollout_trace(trace_records_or_path)
    return [dict(record) for record in trace_records_or_path]


def _trace_arrays(records: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Convert required trace fields into aligned numeric arrays."""
    if not records:
        message = "rollout trace contains no records"
        raise ValueError(message)
    time = np.asarray([record["time_sec"] for record in records], dtype=float)
    episode_index = np.asarray([record.get("episode_index", 0) for record in records], dtype=int)
    actual = np.asarray([_position_row(record, "actual_position_xyz_m", "current_position") for record in records], dtype=float)
    reference = np.asarray([_position_row(record, "reference_position_xyz_m", "reference_position") for record in records], dtype=float)
    reported_position_error = np.asarray([record["position_error_m"] for record in records], dtype=float)
    if actual.shape != reference.shape or actual.ndim != TRACE_POSITION_NDIM or actual.shape[1] != XYZ_DIMENSIONS:
        message = "trace position fields must have shape (steps, 3)"
        raise ValueError(message)
    position_error = np.linalg.norm(actual - reference, axis=1)
    if time.shape != reported_position_error.shape or time.shape[0] != actual.shape[0]:
        message = "trace time and position_error fields must align with positions"
        raise ValueError(message)
    if not np.allclose(reported_position_error, position_error, atol=1.0e-9, rtol=1.0e-9):
        message = "trace position_error_m must equal same-row actual/reference position distance"
        raise ValueError(message)
    if not all(np.all(np.isfinite(array)) for array in (time, actual, reference, position_error)):
        message = "trace plot fields must contain only finite values"
        raise ValueError(message)
    for active_episode in np.unique(episode_index):
        episode_time = time[episode_index == active_episode]
        if episode_time.shape[0] > 1 and np.any(np.diff(episode_time) < 0.0):
            message = "trace time_sec must be monotonic within each episode"
            raise ValueError(message)
    return {
        "time": time,
        "episode_index": episode_index,
        "actual": actual,
        "reference": reference,
        "position_error": position_error,
        "start_hold_enabled": np.asarray([bool(record.get("start_hold_enabled", False)) for record in records], dtype=bool),
        "tracking_phase_start_time_sec": np.asarray(
            [float(record.get("tracking_phase_start_time_sec", 0.0)) for record in records],
            dtype=float,
        ),
    }


def _position_row(record: Mapping[str, Any], primary_key: str, fallback_key: str) -> np.ndarray:
    """Return one finite XYZ position row from a trace record."""
    value = record.get(primary_key, record.get(fallback_key))
    row = np.asarray(value, dtype=float).reshape(-1)
    if row.shape != (XYZ_DIMENSIONS,):
        message = f"trace field {primary_key} must contain exactly {XYZ_DIMENSIONS} values"
        raise ValueError(message)
    if not np.all(np.isfinite(row)):
        message = f"trace field {primary_key} must contain only finite values"
        raise ValueError(message)
    return row


def _episode_segments(arrays: Mapping[str, np.ndarray]) -> list[np.ndarray]:
    """Return contiguous row indices grouped by episode to avoid reset-spanning lines."""
    episode_index = np.asarray(arrays["episode_index"], dtype=int)
    segments: list[np.ndarray] = []
    start = 0
    for index in range(1, episode_index.shape[0]):
        if episode_index[index] == episode_index[index - 1]:
            continue
        segments.append(np.arange(start, index))
        start = index
    segments.append(np.arange(start, episode_index.shape[0]))
    return segments


def _mark_tracking_starts(
    axis: Any,
    arrays: Mapping[str, np.ndarray],
    segments: Sequence[np.ndarray],
    label: bool,
) -> None:
    """Draw tracking-start markers on time plots when start-hold metadata is present."""
    labeled = False
    for segment in segments:
        if not np.any(arrays["start_hold_enabled"][segment]):
            continue
        start_time = float(arrays["tracking_phase_start_time_sec"][segment][0])
        if not np.isfinite(start_time):
            continue
        segment_times = arrays["time"][segment]
        if segment_times.size == 0 or start_time < float(np.min(segment_times)) or start_time > float(np.max(segment_times)):
            continue
        axis.axvline(
            start_time,
            color="#555555",
            linestyle=":",
            linewidth=1.2,
            label="tracking start" if label and not labeled else None,
        )
        labeled = True


def _optional_action_array(records: Sequence[Mapping[str, Any]], key: str) -> np.ndarray | None:
    """Return a flattened action array when every record contains finite action data."""
    action_rows: list[np.ndarray] = []
    for record in records:
        if key not in record or record[key] is None:
            return None
        action = np.asarray(record[key], dtype=float).reshape(-1)
        if action.size == 0 or not np.all(np.isfinite(action)):
            return None
        action_rows.append(action)
    first_size = action_rows[0].shape[0]
    if any(row.shape[0] != first_size for row in action_rows):
        return None
    return np.vstack(action_rows)


def _write_json_fallback(
    reference: trajectories.primitives.Trajectory,
    actual: trajectories.primitives.Trajectory,
    errors: np.ndarray,
    output_path: Path,
) -> TrajectoryPlotResult:
    """Write compact JSON fallback comparison data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "output_kind": "json_fallback",
        "sample_count": int(errors.shape[0]),
        "times": [float(value) for value in reference.times],
        "reference_positions": np.asarray(reference.positions, dtype=float).tolist(),
        "actual_positions": np.asarray(actual.positions, dtype=float).tolist(),
        "position_errors_m": [float(value) for value in errors],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TrajectoryPlotResult(output_path=str(output_path), output_kind="json_fallback", sample_count=int(errors.shape[0]))


__all__ = [
    "RolloutTracePlotResult",
    "TrajectoryPlotResult",
    "write_policy_rollout_trace_plots",
    "write_trajectory_comparison",
]
