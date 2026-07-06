"""Tests for minimal trajectory comparison output helpers."""

# ruff: noqa: S101

from __future__ import annotations

import importlib.util
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from src import evaluation, trajectories

if TYPE_CHECKING:
    from pathlib import Path


def _trajectory_pair() -> tuple[trajectories.primitives.Trajectory, trajectories.primitives.Trajectory]:
    """Return comparable reference and actual trajectories for plot tests."""
    reference = trajectories.primitives.make_line_trajectory(
        start=(0.0, 0.0, 1.0),
        end=(1.0, 0.0, 1.0),
        duration_sec=1.0,
        sample_rate_hz=4.0,
    )
    actual = trajectories.primitives.Trajectory(
        times=np.array(reference.times, dtype=float, copy=True),
        positions=np.array(reference.positions + np.array([0.1, 0.0, 0.0]), dtype=float, copy=True),
    )
    return reference, actual


def test_matplotlib_plot_writer_creates_nonempty_file_when_available(tmp_path: Path) -> None:
    """Verify matplotlib output is created when matplotlib can be imported."""
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib is not available")
    reference, actual = _trajectory_pair()
    output_path = tmp_path / "trajectory_comparison.png"

    result = evaluation.plots.write_trajectory_comparison(reference, actual, output_path)

    assert result.output_path == str(output_path)
    assert result.output_kind == "matplotlib_png"
    assert result.sample_count == reference.times.shape[0]
    assert output_path.stat().st_size > 0


def test_fallback_writer_creates_json_without_matplotlib_dependency(tmp_path: Path) -> None:
    """Verify fallback output can be forced without altering installed packages."""
    reference, actual = _trajectory_pair()
    output_path = tmp_path / "trajectory_comparison.json"

    result = evaluation.plots.write_trajectory_comparison(reference, actual, output_path, force_fallback=True)
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert result.output_kind == "json_fallback"
    assert payload["sample_count"] == reference.times.shape[0]
    assert len(payload["position_errors_m"]) == reference.times.shape[0]


def test_malformed_trajectory_raises_before_writing(tmp_path: Path) -> None:
    """Verify invalid trajectory shapes are rejected before output is written."""
    reference, _actual = _trajectory_pair()
    malformed = trajectories.primitives.Trajectory(
        times=np.array(reference.times, dtype=float, copy=True),
        positions=np.ones((reference.times.shape[0], 2), dtype=float),
    )
    output_path = tmp_path / "bad.png"

    with pytest.raises(ValueError, match="positions"):
        evaluation.plots.write_trajectory_comparison(reference, malformed, output_path)

    assert not output_path.exists()


def test_plots_import_through_package_alias() -> None:
    """Verify plot helpers are exposed by the evaluation package."""
    assert evaluation.plots.write_trajectory_comparison is not None
