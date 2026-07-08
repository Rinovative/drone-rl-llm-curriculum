"""Tests for minimal trajectory comparison output helpers."""

# ruff: noqa: S101

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src import evaluation, trajectories

TRACE_STEP_COUNT = 3


def _trace_records(include_action: bool = True) -> list[dict[str, object]]:
    """Return a small trained-policy trace for plot tests."""
    records: list[dict[str, object]] = []
    for step_index in range(TRACE_STEP_COUNT):
        action: object = [[float(step_index), 0.0, 1.0]] if include_action else None
        records.append(
            {
                "step_index": step_index,
                "time_sec": float(step_index) * 0.1,
                "reward": -0.1,
                "position_error_m": 0.1,
                "actual_position_xyz_m": [0.1 * step_index, 0.0, 1.0],
                "reference_position_xyz_m": [0.1 * step_index, 0.1, 1.0],
                "error_xyz_m": [0.0, -0.1, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "roll_pitch_yaw": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
                "action": action,
                "applied_action": action,
                "terminated": False,
                "truncated": False,
                "termination_reason": "running",
            }
        )
    return records


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


def test_policy_rollout_trace_plots_create_required_pngs(tmp_path: Path) -> None:
    """Verify trained-policy trace plots are generated from saved JSONL traces."""
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib is not available")
    trace_path = tmp_path / "trace.jsonl"
    plots_dir = tmp_path / "plots"
    evaluation.rollout.write_policy_rollout_trace(_trace_records(include_action=True), trace_path)

    result = evaluation.plots.write_policy_rollout_trace_plots(trace_path, plots_dir)

    assert result.output_kind == "matplotlib_png"
    assert result.step_count == TRACE_STEP_COUNT
    assert set(result.plot_paths) == {
        "trajectory_xy",
        "trajectory_xyz",
        "position_error",
        "action_trace",
    }
    for output_path in result.plot_paths.values():
        assert Path(output_path).stat().st_size > 0


def test_policy_rollout_trace_plots_omit_action_plot_without_action_data(tmp_path: Path) -> None:
    """Verify action plots are optional when trace action data is absent."""
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib is not available")

    result = evaluation.plots.write_policy_rollout_trace_plots(_trace_records(include_action=False), tmp_path / "plots")

    assert "action_trace" not in result.plot_paths
    assert "trajectory_xy" in result.plot_paths


def test_policy_rollout_trace_plots_reject_mismatched_error(tmp_path: Path) -> None:
    """Verify plot consistency checks catch same-row error mismatches."""
    records = _trace_records(include_action=False)
    records[0]["position_error_m"] = 42.0

    with pytest.raises(ValueError, match="same-row actual/reference"):
        evaluation.plots.write_policy_rollout_trace_plots(records, tmp_path / "plots")


def test_trace_plot_data_uses_global_step_axis_for_multi_episode_time_resets() -> None:
    """Verify multi-episode local time resets are plotted on a global evaluation axis."""
    records = _trace_records(include_action=False) + _trace_records(include_action=False)
    for step_index, record in enumerate(records):
        record["step_index"] = step_index
        record["episode_index"] = 0 if step_index < TRACE_STEP_COUNT else 1
        record["episode_step_index"] = step_index % TRACE_STEP_COUNT
        record["time_sec"] = float(step_index % TRACE_STEP_COUNT) * 0.1
        record["start_hold_enabled"] = True
        record["is_tracking_phase"] = (step_index % TRACE_STEP_COUNT) >= 1
        record["tracking_phase_start_time_sec"] = 0.1

    plot_data = evaluation.plots._prepare_trace_plot_data(evaluation.plots._trace_arrays(records))  # noqa: SLF001

    assert plot_data.uses_global_step_axis is True
    assert plot_data.x_label == "evaluation step"
    assert plot_data.x.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert [segment.tolist() for segment in plot_data.segments] == [[0, 1, 2], [3, 4, 5]]
    assert plot_data.reset_boundary_x == (3.0,)
    assert plot_data.tracking_start_x == (1.0, 4.0)


def test_trace_plot_data_splits_detected_time_reset_without_episode_index_change() -> None:
    """Verify reset-like time drops split trace segments even without episode metadata."""
    records = _trace_records(include_action=False) + _trace_records(include_action=False)
    for step_index, record in enumerate(records):
        record["step_index"] = step_index
        record["time_sec"] = float(step_index % TRACE_STEP_COUNT) * 0.1

    plot_data = evaluation.plots._prepare_trace_plot_data(evaluation.plots._trace_arrays(records))  # noqa: SLF001

    assert plot_data.uses_global_step_axis is True
    assert [segment.tolist() for segment in plot_data.segments] == [[0, 1, 2], [3, 4, 5]]
    assert plot_data.reset_boundary_x == (3.0,)


def test_trace_plot_data_keeps_single_episode_time_axis_unchanged() -> None:
    """Verify single-episode traces keep the original local time axis."""
    records = _trace_records(include_action=False)

    plot_data = evaluation.plots._prepare_trace_plot_data(evaluation.plots._trace_arrays(records))  # noqa: SLF001

    assert plot_data.uses_global_step_axis is False
    assert plot_data.x_label == "time s"
    assert plot_data.x.tolist() == [0.0, 0.1, 0.2]
    assert [segment.tolist() for segment in plot_data.segments] == [[0, 1, 2]]
    assert plot_data.reset_boundary_x == ()


def test_plots_import_through_package_alias() -> None:
    """Verify plot helpers are exposed by the evaluation package."""
    assert evaluation.plots.write_trajectory_comparison is not None
