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
    resolved_dir = Path(output_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plot_paths: dict[str, str] = {}

    xy_path = resolved_dir / "xy_reference_vs_actual.png"
    figure, axis = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    axis.plot(arrays["reference"][:, 0], arrays["reference"][:, 1], color="#1f77b4", linewidth=2.0, label="reference")
    axis.plot(arrays["actual"][:, 0], arrays["actual"][:, 1], color="#d62728", linewidth=2.0, label="actual")
    axis.set_xlabel("x m")
    axis.set_ylabel("y m")
    axis.set_title("XY reference vs actual")
    axis.axis("equal")
    axis.legend()
    figure.savefig(xy_path, dpi=140)
    plt.close(figure)
    plot_paths["xy_reference_vs_actual"] = str(xy_path)

    xyz_path = resolved_dir / "xyz_vs_time.png"
    figure, axes = plt.subplots(3, 1, figsize=(7.0, 6.0), sharex=True, constrained_layout=True)
    labels = ("x", "y", "z")
    for axis_index, axis in enumerate(axes):
        axis.plot(arrays["time"], arrays["reference"][:, axis_index], color="#1f77b4", linewidth=1.8, label="reference")
        axis.plot(arrays["time"], arrays["actual"][:, axis_index], color="#d62728", linewidth=1.6, label="actual")
        axis.set_ylabel(f"{labels[axis_index]} m")
    axes[0].set_title("XYZ vs time")
    axes[-1].set_xlabel("time s")
    axes[0].legend()
    figure.savefig(xyz_path, dpi=140)
    plt.close(figure)
    plot_paths["xyz_vs_time"] = str(xyz_path)

    error_path = resolved_dir / "position_error_vs_time.png"
    figure, axis = plt.subplots(figsize=(7.0, 3.5), constrained_layout=True)
    axis.plot(arrays["time"], arrays["position_error"], color="#d62728", linewidth=2.0)
    axis.set_xlabel("time s")
    axis.set_ylabel("position error m")
    axis.set_title("Position error vs time")
    figure.savefig(error_path, dpi=140)
    plt.close(figure)
    plot_paths["position_error_vs_time"] = str(error_path)

    action_array = _optional_action_array(records, key="action")
    if action_array is not None:
        action_path = resolved_dir / "action_vs_time.png"
        figure, axis = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
        for action_index in range(action_array.shape[1]):
            axis.plot(arrays["time"], action_array[:, action_index], linewidth=1.5, label=f"action {action_index}")
        axis.set_xlabel("time s")
        axis.set_ylabel("action")
        axis.set_title("Action vs time")
        axis.legend()
        figure.savefig(action_path, dpi=140)
        plt.close(figure)
        plot_paths["action_vs_time"] = str(action_path)

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
    actual = np.asarray([record["actual_position_xyz_m"] for record in records], dtype=float)
    reference = np.asarray([record["reference_position_xyz_m"] for record in records], dtype=float)
    position_error = np.asarray([record["position_error_m"] for record in records], dtype=float)
    if actual.shape != reference.shape or actual.ndim != TRACE_POSITION_NDIM or actual.shape[1] != XYZ_DIMENSIONS:
        message = "trace position fields must have shape (steps, 3)"
        raise ValueError(message)
    if time.shape != position_error.shape or time.shape[0] != actual.shape[0]:
        message = "trace time and position_error fields must align with positions"
        raise ValueError(message)
    if not all(np.all(np.isfinite(array)) for array in (time, actual, reference, position_error)):
        message = "trace plot fields must contain only finite values"
        raise ValueError(message)
    return {
        "time": time,
        "actual": actual,
        "reference": reference,
        "position_error": position_error,
    }


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
