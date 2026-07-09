"""Tests for the scenario/OOD re-evaluation helper script."""

# ruff: noqa: S101

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path("scripts/reevaluate_scenarios.sh")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one test manifest JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _create_direct_run(runs_root: Path) -> None:
    """Create a minimal direct PPO run manifest with a final model."""
    run_root = runs_root / "direct_ppo_pid_dynprev_m-taskdist_medium_seed0"
    model_path = run_root / "training" / "models" / "policy.zip"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"model")
    _write_json(
        run_root / "run_manifest.json",
        {
            "run_name": run_root.name,
            "run_kind": "direct_ppo",
            "training": {"last_model_path_relative": "training/models/policy.zip"},
        },
    )


def _create_curriculum_run(runs_root: Path) -> None:
    """Create a minimal curriculum run manifest with a final-stage model."""
    run_root = runs_root / "curriculum_manual_pid_dynprev_m-taskdist_medium_seed0"
    model_path = run_root / "stages" / "stage05_medium_tracking" / "training" / "models" / "policy.zip"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"model")
    _write_json(
        run_root / "run_manifest.json",
        {
            "run_name": run_root.name,
            "run_kind": "curriculum",
            "curriculum_kind": "manual",
            "stages": [
                {
                    "stage_index": 5,
                    "stage_name": "medium_tracking",
                    "run_name": "curriculum_manual_pid_dynprev_m-taskdist_medium_stage05_medium_tracking_seed0",
                    "last_model_path_relative": "stages/stage05_medium_tracking/training/models/policy.zip",
                }
            ],
        },
    )


def _run_dry_script(runs_root: Path, *, filter_text: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run the scenario script in dry-run mode against a temporary runs root."""
    env = {
        **os.environ,
        "PYTHON_BIN": sys.executable,
        "RUNS_ROOT": str(runs_root),
        "DRY_RUN": "1",
    }
    if filter_text is not None:
        env["FILTER"] = filter_text
    return subprocess.run(  # noqa: S603
        [str(SCRIPT_PATH)],
        check=True,
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
    )


def test_reevaluate_scenarios_dry_run_prints_valid_commands(tmp_path: Path) -> None:
    """Verify dry-run prints scenario-only commands without malformed YAML arguments."""
    runs_root = tmp_path / "runs"
    _create_direct_run(runs_root)
    _create_curriculum_run(runs_root)

    result = _run_dry_script(runs_root)

    assert "Scenario/OOD re-evaluation dry run complete." in result.stdout
    assert "experiments_cli_evaluate_policy" in result.stdout
    assert "experiments_cli_evaluate_curriculum" in result.stdout
    assert "--profile scenario" in result.stdout
    assert "--model-scope final-stage" in result.stdout
    assert ".yaml--" not in result.stdout


def test_reevaluate_scenarios_filter_curriculum_selects_curriculum_runs(tmp_path: Path) -> None:
    """Verify FILTER=curriculum narrows dry-run commands to curriculum runs."""
    runs_root = tmp_path / "runs"
    _create_direct_run(runs_root)
    _create_curriculum_run(runs_root)

    result = _run_dry_script(runs_root, filter_text="curriculum")

    assert "curriculum_manual_pid_dynprev_m-taskdist_medium_seed0" in result.stdout
    assert "direct_ppo_pid_dynprev_m-taskdist_medium_seed0" not in result.stdout
    assert "experiments_cli_evaluate_curriculum" in result.stdout
    assert "experiments_cli_evaluate_policy" not in result.stdout


def test_reevaluate_scenarios_filter_direct_ppo_selects_direct_runs(tmp_path: Path) -> None:
    """Verify FILTER=direct_ppo narrows dry-run commands to direct PPO runs."""
    runs_root = tmp_path / "runs"
    _create_direct_run(runs_root)
    _create_curriculum_run(runs_root)

    result = _run_dry_script(runs_root, filter_text="direct_ppo")

    assert "direct_ppo_pid_dynprev_m-taskdist_medium_seed0" in result.stdout
    assert "curriculum_manual_pid_dynprev_m-taskdist_medium_seed0" not in result.stdout
    assert "experiments_cli_evaluate_policy" in result.stdout
    assert "experiments_cli_evaluate_curriculum" not in result.stdout
