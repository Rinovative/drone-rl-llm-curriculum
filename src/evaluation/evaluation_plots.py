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
    "real_action_trace": "real_action_trace.png",
}

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from src import trajectories


@dataclass(frozen=True)
class _TracePlotData:
    """Prepared arrays and segment metadata for one policy trace plot set."""

    arrays: dict[str, np.ndarray]
    segments: list[np.ndarray]
    x: np.ndarray
    x_label: str
    episode_count: int
    uses_global_step_axis: bool
    reset_boundary_x: tuple[float, ...]
    tracking_start_x: tuple[float, ...]


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
    plot_data = _prepare_trace_plot_data(arrays)
    segments = plot_data.segments
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
            label=_trace_line_label("reference", plot_data) if segment_index == 0 else None,
        )
        axis.plot(
            arrays["actual"][segment, 0],
            arrays["actual"][segment, 1],
            color="#d62728",
            linewidth=2.0,
            label=_trace_line_label("actual", plot_data) if segment_index == 0 else None,
        )
    _mark_xy_initial_positions(axis, arrays)
    axis.set_xlabel("x m")
    axis.set_ylabel("y m")
    axis.set_title(_trace_plot_title("XY reference vs actual", plot_data))
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
                plot_data.x[segment],
                arrays["reference"][segment, axis_index],
                color="#1f77b4",
                linewidth=1.8,
                label="reference" if axis_index == 0 and segment_index == 0 else None,
            )
            axis.plot(
                plot_data.x[segment],
                arrays["actual"][segment, axis_index],
                color="#d62728",
                linewidth=1.6,
                label="actual" if axis_index == 0 and segment_index == 0 else None,
            )
        _mark_episode_boundaries(axis, plot_data=plot_data, label=axis_index == 0)
        _mark_tracking_starts(axis, plot_data=plot_data, label=axis_index == 0)
        axis.set_ylabel(f"{labels[axis_index]} m")
    axes[0].set_title(_trace_plot_title("XYZ trajectory components", plot_data))
    axes[-1].set_xlabel(plot_data.x_label)
    axes[0].legend()
    figure.savefig(xyz_path, dpi=140)
    plt.close(figure)
    plot_paths["trajectory_xyz"] = str(xyz_path)

    error_path = resolved_dir / CANONICAL_POLICY_PLOT_FILENAMES["position_error"]
    figure, axis = plt.subplots(figsize=(7.0, 3.5), constrained_layout=True)
    for segment in segments:
        axis.plot(plot_data.x[segment], arrays["position_error"][segment], color="#d62728", linewidth=2.0)
    _mark_episode_boundaries(axis, plot_data=plot_data, label=True)
    _mark_tracking_starts(axis, plot_data=plot_data, label=True)
    axis.set_xlabel(plot_data.x_label)
    axis.set_ylabel("position error m")
    axis.set_title(_trace_plot_title("Position error", plot_data))
    figure.savefig(error_path, dpi=140)
    plt.close(figure)
    plot_paths["position_error"] = str(error_path)

    action_array = _optional_action_array(records, key="action")
    if action_array is not None:
        action_path = resolved_dir / CANONICAL_POLICY_PLOT_FILENAMES["action_trace"]
        _write_action_trace_plot(
            plt=plt,
            path=action_path,
            records=records,
            action_array=action_array,
            plot_data=plot_data,
            action_key="action",
        )
        plot_paths["action_trace"] = str(action_path)

    real_action_array = _optional_action_array(records, key="real_action")
    if real_action_array is not None:
        real_action_path = resolved_dir / CANONICAL_POLICY_PLOT_FILENAMES["real_action_trace"]
        _write_action_trace_plot(
            plt=plt,
            path=real_action_path,
            records=records,
            action_array=real_action_array,
            plot_data=plot_data,
            action_key="real_action",
        )
        plot_paths["real_action_trace"] = str(real_action_path)

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
    step_index = np.asarray([record.get("step_index", index) for index, record in enumerate(records)], dtype=float)
    episode_step_index = np.asarray([record.get("episode_step_index", index) for index, record in enumerate(records)], dtype=int)
    episode_index = np.asarray([record.get("episode_index", 0) for record in records], dtype=int)
    actual = np.asarray([_position_row(record, "actual_position_xyz_m", "current_position") for record in records], dtype=float)
    reference = np.asarray([_position_row(record, "reference_position_xyz_m", "reference_position") for record in records], dtype=float)
    reported_position_error = np.asarray([record["position_error_m"] for record in records], dtype=float)
    if actual.shape != reference.shape or actual.ndim != TRACE_POSITION_NDIM or actual.shape[1] != XYZ_DIMENSIONS:
        message = "trace position fields must have shape (steps, 3)"
        raise ValueError(message)
    position_error = np.linalg.norm(actual - reference, axis=1)
    if time.shape != reported_position_error.shape or time.shape[0] != actual.shape[0] or step_index.shape[0] != actual.shape[0]:
        message = "trace time, step_index, and position_error fields must align with positions"
        raise ValueError(message)
    if not np.allclose(reported_position_error, position_error, atol=1.0e-9, rtol=1.0e-9):
        message = "trace position_error_m must equal same-row actual/reference position distance"
        raise ValueError(message)
    if not all(np.all(np.isfinite(array)) for array in (time, step_index, actual, reference, position_error)):
        message = "trace plot fields must contain only finite values"
        raise ValueError(message)
    return {
        "time": time,
        "step_index": step_index,
        "episode_step_index": episode_step_index,
        "episode_index": episode_index,
        "actual": actual,
        "reference": reference,
        "position_error": position_error,
        "start_hold_enabled": np.asarray([bool(record.get("start_hold_enabled", False)) for record in records], dtype=bool),
        "is_tracking_phase": np.asarray([bool(record.get("is_tracking_phase", False)) for record in records], dtype=bool),
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


def _prepare_trace_plot_data(arrays: dict[str, np.ndarray]) -> _TracePlotData:
    """Return segment metadata and the x-axis that makes episode resets explicit."""
    segments = _episode_segments(arrays)
    episode_count = int(np.unique(arrays["episode_index"]).shape[0])
    use_global_step_axis = episode_count > 1 or len(segments) > 1
    if use_global_step_axis:
        x = _global_step_x(arrays["step_index"])
        x_label = "evaluation step"
    else:
        x = np.asarray(arrays["time"], dtype=float)
        x_label = "time s"
    reset_boundary_x = tuple(float(x[segment[0]]) for segment in segments[1:] if segment.size > 0)
    tracking_start_x = _tracking_start_x_values(arrays=arrays, segments=segments, x=x)
    return _TracePlotData(
        arrays=arrays,
        segments=segments,
        x=x,
        x_label=x_label,
        episode_count=episode_count,
        uses_global_step_axis=use_global_step_axis,
        reset_boundary_x=reset_boundary_x,
        tracking_start_x=tracking_start_x,
    )


def _global_step_x(step_index: np.ndarray) -> np.ndarray:
    """Return a nondecreasing global x-axis for multi-episode time-series plots."""
    x = np.asarray(step_index, dtype=float)
    if x.shape[0] > 1 and np.any(np.diff(x) < 0.0):
        return np.arange(x.shape[0], dtype=float)
    return x


def _episode_segments(arrays: Mapping[str, np.ndarray]) -> list[np.ndarray]:
    """Return contiguous row indices grouped by episode or detected time reset."""
    episode_index = np.asarray(arrays["episode_index"], dtype=int)
    time = np.asarray(arrays["time"], dtype=float)
    segments: list[np.ndarray] = []
    start = 0
    for index in range(1, episode_index.shape[0]):
        same_episode = episode_index[index] == episode_index[index - 1]
        time_did_not_reset = time[index] >= time[index - 1]
        if same_episode and time_did_not_reset:
            continue
        segments.append(np.arange(start, index))
        start = index
    segments.append(np.arange(start, episode_index.shape[0]))
    return segments


def _tracking_start_x_values(arrays: Mapping[str, np.ndarray], segments: Sequence[np.ndarray], x: np.ndarray) -> tuple[float, ...]:
    """Return per-episode tracking-start marker positions on the selected x-axis."""
    start_values: list[float] = []
    for segment in segments:
        if segment.size == 0 or not np.any(arrays["start_hold_enabled"][segment]):
            continue
        tracking_rows = segment[arrays["is_tracking_phase"][segment]]
        if tracking_rows.size > 0:
            start_values.append(float(x[tracking_rows[0]]))
            continue
        start_time = float(arrays["tracking_phase_start_time_sec"][segment][0])
        if not np.isfinite(start_time):
            continue
        candidate_rows = segment[arrays["time"][segment] >= start_time]
        if candidate_rows.size > 0:
            start_values.append(float(x[candidate_rows[0]]))
    return tuple(start_values)


def _mark_episode_boundaries(axis: Any, plot_data: _TracePlotData, label: bool) -> None:
    """Draw reset markers on time-series plots for multi-episode traces."""
    labeled = False
    for reset_x in plot_data.reset_boundary_x:
        axis.axvline(
            reset_x,
            color="#888888",
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
            label="episode reset" if label and not labeled else None,
        )
        labeled = True


def _mark_tracking_starts(axis: Any, plot_data: _TracePlotData, label: bool) -> None:
    """Draw tracking-start markers on time plots when start-hold metadata is present."""
    labeled = False
    for start_x in plot_data.tracking_start_x:
        axis.axvline(
            start_x,
            color="#555555",
            linestyle=":",
            linewidth=1.2,
            label="tracking start" if label and not labeled else None,
        )
        labeled = True


def _mark_xy_initial_positions(axis: Any, arrays: dict[str, np.ndarray]) -> None:
    """Draw actual and reference start markers on the XY trace plot."""
    reference_start = arrays["reference"][0, :2]
    actual_start = arrays["actual"][0, :2]
    axis.scatter(
        reference_start[0],
        reference_start[1],
        color="#1f77b4",
        marker="o",
        s=45,
        linewidths=0.8,
        edgecolors="white",
        label="reference start",
        zorder=5,
    )
    axis.scatter(
        actual_start[0],
        actual_start[1],
        color="#d62728",
        marker="x",
        s=55,
        linewidths=1.6,
        label="actual start",
        zorder=6,
    )


def _write_action_trace_plot(
    plt: Any,
    path: Path,
    records: Sequence[Mapping[str, Any]],
    action_array: np.ndarray,
    plot_data: _TracePlotData,
    action_key: str,
) -> None:
    """Write one action trace plot with labels derived from action metadata."""
    figure, axis = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    labels = _action_line_labels(records, action_key=action_key, dimension_count=action_array.shape[1])
    for action_index in range(action_array.shape[1]):
        for segment_index, segment in enumerate(plot_data.segments):
            axis.plot(
                plot_data.x[segment],
                action_array[segment, action_index],
                linewidth=1.5,
                label=labels[action_index] if segment_index == 0 else None,
            )
    if action_key == "real_action" and _trace_action_interface(records) == "pid_position" and action_array.shape[1] >= XYZ_DIMENSIONS:
        _plot_pid_real_action_z_context(axis=axis, plot_data=plot_data)
    _mark_episode_boundaries(axis, plot_data=plot_data, label=True)
    _mark_tracking_starts(axis, plot_data=plot_data, label=True)
    axis.set_xlabel(plot_data.x_label)
    axis.set_ylabel(_action_y_label(records, action_key=action_key))
    axis.set_title(_trace_plot_title(_action_plot_title(records, action_key=action_key), plot_data))
    axis.legend()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_pid_real_action_z_context(axis: Any, plot_data: _TracePlotData) -> None:
    """Overlay current/reference z on real PID target plots for altitude audits."""
    arrays = plot_data.arrays
    for segment_index, segment in enumerate(plot_data.segments):
        axis.plot(
            plot_data.x[segment],
            arrays["reference"][segment, 2],
            color="#1f77b4",
            linestyle="--",
            linewidth=1.2,
            label="reference z [m]" if segment_index == 0 else None,
        )
        axis.plot(
            plot_data.x[segment],
            arrays["actual"][segment, 2],
            color="#d62728",
            linestyle=":",
            linewidth=1.2,
            label="current z [m]" if segment_index == 0 else None,
        )


def _action_line_labels(records: Sequence[Mapping[str, Any]], action_key: str, dimension_count: int) -> list[str]:
    """Return action legend labels from trace metadata."""
    interface = _trace_action_interface(records)
    real_action_type = _trace_real_action_type(records)
    if interface == "pid_position" and dimension_count == XYZ_DIMENSIONS:
        if action_key == "real_action" or (real_action_type == "pid_target_position" and not _trace_actions_normalized(records)):
            return ["real x target [m]", "real y target [m]", "real z target [m]"]
        return ["norm x target", "norm y target", "norm z target"]
    if interface == "direct_rpm":
        prefix = (
            "real motor"
            if action_key == "real_action" or (real_action_type == "motor_rpm" and not _trace_actions_normalized(records))
            else "norm motor"
        )
        suffix = " [rpm]" if prefix == "real motor" else ""
        return [f"{prefix} {index}{suffix}" for index in range(dimension_count)]
    if action_key == "real_action" and real_action_type == "motor_rpm":
        return [f"real motor {index} [rpm]" for index in range(dimension_count)]
    return [f"action {index}" for index in range(dimension_count)]


def _action_y_label(records: Sequence[Mapping[str, Any]], action_key: str) -> str:
    """Return a y-axis label for PPO-facing or real action plots."""
    if action_key == "real_action":
        if _trace_real_action_type(records) == "motor_rpm":
            return "motor rpm"
        if _trace_real_action_type(records) == "pid_target_position":
            return "target position m"
        return "real action"
    return "normalized action" if _trace_actions_normalized(records) else "action"


def _action_plot_title(records: Sequence[Mapping[str, Any]], action_key: str) -> str:
    """Return a title that names the plotted action representation."""
    if action_key == "real_action":
        real_action_type = _trace_real_action_type(records)
        if real_action_type == "pid_target_position":
            return "Real PID target trace"
        if real_action_type == "motor_rpm":
            return "Real motor RPM trace"
        return "Real action trace"
    interface = _trace_action_interface(records)
    if interface == "pid_position" and _trace_actions_normalized(records):
        return "Normalized PID target action trace"
    if interface == "direct_rpm":
        return "Normalized direct-RPM action trace"
    return "Action trace"


def _trace_action_interface(records: Sequence[Mapping[str, Any]]) -> str:
    """Return the action interface recorded in a trace, if present."""
    return str(records[0].get("action_interface", "")) if records else ""


def _trace_real_action_type(records: Sequence[Mapping[str, Any]]) -> str:
    """Return the real action type recorded in a trace, if present."""
    return str(records[0].get("real_action_type", "")) if records else ""


def _trace_actions_normalized(records: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether trace actions are explicitly marked as normalized."""
    return bool(records and records[0].get("actions_normalized", False))


def _trace_plot_title(base_title: str, plot_data: _TracePlotData) -> str:
    """Return a plot title that makes multi-episode traces explicit."""
    if plot_data.episode_count > 1:
        return f"{base_title} ({plot_data.episode_count} evaluation episodes)"
    if len(plot_data.segments) > 1:
        return f"{base_title} ({len(plot_data.segments)} reset segments)"
    return base_title


def _trace_line_label(label: str, plot_data: _TracePlotData) -> str:
    """Return legend text that distinguishes episode segments from stages."""
    if plot_data.episode_count > 1 or len(plot_data.segments) > 1:
        return f"{label} episode segments"
    return label


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
