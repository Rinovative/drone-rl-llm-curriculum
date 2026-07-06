"""Tests for trained PPO policy render helpers."""

# ruff: noqa: S101

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src import experiments
from src.experiments import cli_render_policy

PARSER_CAMERA_DISTANCE = 0.95
PARSER_CAMERA_YAW = 33.0
PARSER_CAMERA_PITCH = -18.0
RENDER_MAX_STEPS = 120
RENDER_ACTUAL_STEPS = 118
RENDER_REQUIRED_REFERENCE_SAMPLES = 121


def test_policy_render_imports_through_package_alias() -> None:
    """Verify trained-policy render helpers are exposed by the experiments package."""
    assert experiments.policy_render is not None
    assert experiments.policy_render.PolicyRenderSettings is not None


def test_policy_render_settings_defaults_are_expected() -> None:
    """Verify policy-render defaults stay aligned with reviewer-facing CLI expectations."""
    settings = experiments.policy_render.PolicyRenderSettings()

    assert settings.model_path == Path("storage/models/ppo_tracking_smoke/ppo_tracking_smoke.zip")
    assert settings.config_path == Path("configs/smoke/ppo_tracking_smoke.yaml")
    assert settings.output_dir is None
    assert settings.max_steps == experiments.policy_render.DEFAULT_MAX_STEPS
    assert settings.seed == 0
    assert settings.camera_mode == "follow_external"
    assert settings.camera_distance == experiments.policy_render.DEFAULT_CAMERA_DISTANCE_M


def test_policy_render_settings_reject_invalid_max_steps() -> None:
    """Verify invalid rollout step limits are rejected."""
    with pytest.raises(ValueError, match="max_steps must be positive"):
        experiments.policy_render.PolicyRenderSettings(max_steps=0)


def test_policy_render_settings_reject_invalid_camera_distance() -> None:
    """Verify non-positive camera distances are rejected."""
    with pytest.raises(ValueError, match="camera_distance must be finite and positive"):
        experiments.policy_render.PolicyRenderSettings(camera_distance=0.0)


def test_cli_parser_accepts_camera_and_render_task_options() -> None:
    """Verify parser exposes render-task and camera visibility controls."""
    parser = cli_render_policy.build_parser()
    args = parser.parse_args(
        [
            "--render-task-shape",
            "circle",
            "--camera-distance",
            str(PARSER_CAMERA_DISTANCE),
            "--camera-yaw",
            str(PARSER_CAMERA_YAW),
            "--camera-pitch",
            str(PARSER_CAMERA_PITCH),
        ]
    )

    assert args.render_task_shape == "circle"
    assert args.camera_distance == PARSER_CAMERA_DISTANCE
    assert args.camera_yaw == PARSER_CAMERA_YAW
    assert args.camera_pitch == PARSER_CAMERA_PITCH


def test_prepare_task_for_rollout_length_extends_short_reference() -> None:
    """Verify short tasks are densified so requested rollout lengths are reachable."""
    config = experiments.config.load_experiment_config("configs/smoke/trajectory_validation.yaml")
    line_task = dict(config["tasks"][2])

    prepared_task, reference_samples, warnings = experiments.policy_render._prepare_task_for_rollout_length(  # noqa: SLF001
        task=line_task,
        requested_max_steps=RENDER_MAX_STEPS,
    )

    assert prepared_task["shape"] == "line"
    assert reference_samples >= RENDER_REQUIRED_REFERENCE_SAMPLES
    assert warnings


def test_policy_render_missing_model_path_raises_clear_error(tmp_path: Path) -> None:
    """Verify missing trained model paths fail with a useful training command hint."""
    missing_model = tmp_path / "missing_model.zip"

    with pytest.raises(FileNotFoundError, match="cli_train_tracking"):
        experiments.policy_render.run_trained_policy_render_from_paths(
            model_path=missing_model,
            config_path=Path("configs/smoke/ppo_tracking_smoke.yaml"),
            output_dir=tmp_path,
            max_steps=4,
            seed=0,
        )


def test_policy_render_manifest_includes_rollout_summary_fields() -> None:
    """Verify manifest payload includes requested fields for rollout explainability."""
    settings = experiments.policy_render.PolicyRenderSettings()
    payload = experiments.policy_render._build_manifest(  # noqa: SLF001
        settings=settings,
        model_path=Path("storage/models/ppo_tracking_smoke/ppo_tracking_smoke.zip"),
        gif_path=Path("storage/results/trained_policy_render/trained_policy_rollout.gif"),
        task_shape="line",
        task_source="config",
        task_index=2,
        requested_max_steps=RENDER_MAX_STEPS,
        actual_steps=RENDER_ACTUAL_STEPS,
        termination_reason="terminated_reference_complete",
        position_bounds={"min": [0.0, 0.0, 1.0], "max": [1.0, 0.0, 1.0]},
        reference_position_bounds={"min": [0.0, 0.0, 1.0], "max": [1.0, 0.0, 1.0]},
        true_simulator_rendering=True,
        policy_predict_used=True,
        metrics={"mode": "trained_policy_render", "steps": RENDER_ACTUAL_STEPS},
        warnings=(),
        final_info={
            "base_terminated": False,
            "base_truncated": False,
            "current_position": [1.0, 0.0, 1.0],
            "reference_position": [1.0, 0.0, 1.0],
            "position_error_m": 0.0,
            "roll_pitch_yaw": [0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
            "last_action": [1.0, 1.0, 1.0, 1.0],
            "base_info_keys": ["answer"],
            "base_reason_fields": {},
            "base_truncation_causes": [],
        },
        final_action=[[0.0, 0.0, 1.0]],
    )

    assert payload["requested_max_steps"] == RENDER_MAX_STEPS
    assert payload["actual_steps"] == RENDER_ACTUAL_STEPS
    assert payload["survived_fraction"] == RENDER_ACTUAL_STEPS / RENDER_MAX_STEPS
    assert payload["termination_reason"] == "terminated_reference_complete"
    assert payload["base_terminated"] is False
    assert payload["base_truncated"] is False
    assert payload["position_bounds"]["max"] == [1.0, 0.0, 1.0]
    assert payload["reference_position_bounds"]["max"] == [1.0, 0.0, 1.0]
    assert payload["z_span_m"] == 0.0
    assert payload["xy_span_m"] == 1.0
    assert payload["final_position"] == [1.0, 0.0, 1.0]
    assert payload["final_reference_position"] == [1.0, 0.0, 1.0]
    assert payload["final_position_error_m"] == 0.0
    assert payload["final_action"] == [[0.0, 0.0, 1.0]]
    assert payload["camera_settings"]["mode"] == settings.camera_mode


def test_policy_render_manifest_writer_writes_json(tmp_path: Path) -> None:
    """Verify the internal manifest writer writes deterministic JSON output."""
    manifest_path = tmp_path / "trained_policy_render_manifest.json"
    rollout_path = tmp_path / "rollout.gif"
    payload = {
        "mode": "trained_policy_render",
        "steps": 3,
        "output_files": [str(rollout_path)],
        "warnings": [],
    }

    result = experiments.policy_render._write_manifest(manifest_path, payload)  # noqa: SLF001
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result == payload
    assert loaded == payload


def test_cli_render_policy_help_works() -> None:
    """Verify the trained-policy render CLI help path works without running rendering."""
    completed = subprocess.run(
        [sys.executable, "-m", "src.experiments.cli_render_policy", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--model-path" in completed.stdout
    assert "--render-task-shape" in completed.stdout
    assert "--camera-distance" in completed.stdout
    assert "--camera-yaw" in completed.stdout
    assert "--camera-pitch" in completed.stdout
