"""Tests for tiny deterministic MVP training-smoke helpers."""

# ruff: noqa: S101

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.experiments.cli import experiments_cli_training_smoke as cli_training_smoke
from src.experiments.training import experiments_training_smoke as training_smoke

CONFIG_MAX_STEPS = 16
TINY_SMOKE_STEPS = 4
PARSER_MAX_STEPS = 3

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_training_smoke_config_loads_default_settings() -> None:
    """Verify the smoke training config loads into validated settings."""
    settings = training_smoke.load_training_smoke_settings("configs/smoke/training_smoke.yaml")

    assert settings.task_config_path.as_posix() == "configs/smoke/trajectory_validation.yaml"
    assert settings.task_index == 0
    assert settings.max_steps == CONFIG_MAX_STEPS
    assert settings.output_filename == "training_smoke_metrics.json"
    assert settings.mode == "deterministic"


def test_deterministic_training_smoke_writes_expected_metrics(tmp_path: Path) -> None:
    """Verify the deterministic fallback runs and writes JSON metrics."""
    settings = training_smoke.TrainingSmokeSettings(output_dir=tmp_path, max_steps=TINY_SMOKE_STEPS)

    result = training_smoke.run_training_smoke(settings)
    output_path = tmp_path / "training_smoke_metrics.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert result.output_path == str(output_path)
    assert payload["mode"] == "deterministic"
    assert payload["baseline"] == "deterministic_offset_decay"
    assert payload["validated"] is True
    assert payload["step_count"] == TINY_SMOKE_STEPS
    assert payload["mean_position_error_m"] >= 0.0
    assert payload["final_position_error_m"] == 0.0
    assert payload["warnings"]


def test_training_smoke_default_output_uses_training_metrics_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify default output paths stay under the canonical training metrics directory."""
    storage_root = tmp_path / "storage"
    monkeypatch.setenv("STORAGE_ROOT", str(storage_root))

    result = training_smoke.run_training_smoke(training_smoke.TrainingSmokeSettings(max_steps=2))

    expected_path = storage_root / "runs" / "mvp_smoke" / "training" / "metrics" / "training_smoke_metrics.json"
    assert result.output_path == str(expected_path)
    assert expected_path.exists()


def test_cli_parser_builds_settings_without_running_training() -> None:
    """Verify parser defaults and overrides are available without side effects."""
    parser = cli_training_smoke.build_parser()
    args = parser.parse_args(["--config", "configs/smoke/training_smoke.yaml", "--max-steps", "3", "--task-index", "1"])

    assert args.config.as_posix() == "configs/smoke/training_smoke.yaml"
    assert args.max_steps == PARSER_MAX_STEPS
    assert args.task_index == 1


def test_cli_main_runs_with_temporary_output_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify the CLI main function runs the bounded deterministic smoke path."""
    status = cli_training_smoke.main(
        [
            "--config",
            "configs/smoke/training_smoke.yaml",
            "--output-dir",
            str(tmp_path),
            "--max-steps",
            "3",
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert "training_smoke_metrics.json" in captured.out
    assert (tmp_path / "training_smoke_metrics.json").exists()
