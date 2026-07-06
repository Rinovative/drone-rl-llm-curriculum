"""
===============================================================================
evaluation_plots.py
===============================================================================
Write minimal visible trajectory comparison outputs for MVP evaluation.

Responsibilities:
  - Validate comparable reference and actual sampled trajectories
  - Write headless matplotlib reference-vs-actual trajectory plots when available
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
from typing import TYPE_CHECKING

import numpy as np

from src import evaluation

if TYPE_CHECKING:
    from src import trajectories


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
    "TrajectoryPlotResult",
    "write_trajectory_comparison",
]
