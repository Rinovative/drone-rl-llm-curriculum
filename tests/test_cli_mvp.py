"""Tests for the MVP reproduction CLI helper."""

# ruff: noqa: S101

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.experiments.cli import experiments_cli_mvp as cli_mvp

DEFAULT_MAX_STEPS = 16

if TYPE_CHECKING:
    from pathlib import Path


def test_mvp_parser_defaults_are_tiny_and_storage_scoped() -> None:
    """Verify parser defaults point to tiny smoke settings."""
    args = cli_mvp.build_parser().parse_args([])

    assert args.config.as_posix() == "tests/fixtures/configs/smoke/training_smoke.yaml"
    assert args.output_dir.as_posix().endswith("storage/runs/mvp_smoke")
    assert args.max_steps == DEFAULT_MAX_STEPS
    assert args.task_index is None
    assert not args.skip_plot


def test_print_commands_mode_outputs_repro_commands_without_artifacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify command-print mode does not create output artifacts."""
    status = cli_mvp.main(["--print-commands", "--output-dir", str(tmp_path), "--max-steps", "5"])

    captured = capsys.readouterr()
    assert status == 0
    assert "python -m src.experiments.cli.experiments_cli_training_smoke" in captured.out
    assert "python -m src.experiments.cli.experiments_cli_mvp" in captured.out
    assert not (tmp_path / "training_smoke_metrics.json").exists()


def test_mvp_main_runs_sequence_with_temporary_output_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify the MVP CLI runs the tiny deterministic sequence."""
    status = cli_mvp.main(
        [
            "--config",
            "tests/fixtures/configs/smoke/training_smoke.yaml",
            "--output-dir",
            str(tmp_path),
            "--max-steps",
            "4",
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert "training_output_path" in captured.out
    assert (tmp_path / "metrics" / "training_smoke_metrics.json").exists()
    assert (tmp_path / "metrics" / "rollout_metrics.json").exists()
    assert (tmp_path / "plots" / "trajectory_comparison.png").exists() or (tmp_path / "plots" / "trajectory_comparison.json").exists()


def test_mvp_main_can_skip_plot(tmp_path: Path) -> None:
    """Verify callers can run metrics-only MVP smoke output."""
    summary = cli_mvp.run_mvp_sequence(output_dir=tmp_path, max_steps=3, skip_plot=True)

    assert summary["visualization"] is None
    assert (tmp_path / "metrics" / "training_smoke_metrics.json").exists()
    assert (tmp_path / "metrics" / "rollout_metrics.json").exists()


def test_missing_prerequisites_raise_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify missing MVP modules produce an actionable error."""
    monkeypatch.delattr(cli_mvp.evaluation, "plots")

    with pytest.raises(RuntimeError, match="missing MVP prerequisites"):
        cli_mvp.run_mvp_sequence(skip_plot=True)
