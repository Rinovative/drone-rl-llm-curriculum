"""Tests for trained PPO policy render helpers."""

# ruff: noqa: S101

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src import envs
from src.experiments import experiments_config
from src.experiments.cli import experiments_cli_render_policy as cli_render_policy
from src.experiments.rendering import experiments_rendering_policy as policy_render

PARSER_CAMERA_DISTANCE = 0.95
PARSER_CAMERA_YAW = 33.0
PARSER_CAMERA_PITCH = -18.0
RENDER_MAX_STEPS = 120
RENDER_ACTUAL_STEPS = 118
RENDER_REQUIRED_REFERENCE_SAMPLES = 121
POLYLINE_RENDER_MAX_STEPS = 200
POLYLINE_REQUIRED_REFERENCE_SAMPLES = 201
LINE_TASK_INDEX = 2
SUPPORTED_RENDER_SHAPES = ("hover", "line", "circle", "polyline")
SCRIPTED_ACTION_STEP_INDEX = 1


def test_policy_render_imports_through_package_alias() -> None:
    """Verify trained-policy render helpers are exposed by the experiments package."""
    assert policy_render is not None
    assert policy_render.PolicyRenderSettings is not None


def test_policy_render_settings_defaults_are_expected() -> None:
    """Verify policy-render defaults stay aligned with reviewer-facing CLI expectations."""
    settings = policy_render.PolicyRenderSettings()

    assert settings.model_path.as_posix().endswith("storage/runs/direct_ppo_hover_seed0/training/models/direct_ppo_hover_seed0.zip")
    assert settings.config_path == Path("configs/training/ppo_tracking.yaml")
    assert settings.output_dir is None
    assert settings.max_steps == policy_render.DEFAULT_MAX_STEPS
    assert settings.seed == 0
    assert settings.camera_mode == "follow_external"
    assert settings.controller == policy_render.PPO_CONTROLLER
    assert settings.model_run_name is None
    assert settings.run_name is None
    assert settings.camera_distance == policy_render.DEFAULT_CAMERA_DISTANCE_M


def test_policy_render_settings_reject_invalid_max_steps() -> None:
    """Verify invalid rollout step limits are rejected."""
    with pytest.raises(ValueError, match="max_steps must be positive"):
        policy_render.PolicyRenderSettings(max_steps=0)


def test_policy_render_settings_reject_invalid_controller() -> None:
    """Verify unsupported rollout controllers are rejected."""
    with pytest.raises(ValueError, match="controller"):
        policy_render.PolicyRenderSettings(controller="hoverboard")


def test_policy_render_settings_reject_invalid_model_run_name() -> None:
    """Verify model run names cannot escape storage/runs."""
    with pytest.raises(ValueError, match="run_name"):
        policy_render.PolicyRenderSettings(model_run_name="../bad")


def test_policy_render_settings_reject_invalid_run_name() -> None:
    """Verify run names cannot escape storage/runs."""
    with pytest.raises(ValueError, match="run_name"):
        policy_render.PolicyRenderSettings(run_name="../bad")


def test_policy_render_settings_reject_invalid_camera_distance() -> None:
    """Verify non-positive camera distances are rejected."""
    with pytest.raises(ValueError, match="camera_distance must be finite and positive"):
        policy_render.PolicyRenderSettings(camera_distance=0.0)


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
    alias_args = parser.parse_args(
        [
            "--task-shape",
            "line",
            "--controller",
            "scripted_reference",
            "--model-run-name",
            "ppo_line_smoke",
            "--run-name",
            "scripted_line",
        ]
    )
    assert alias_args.render_task_shape == "line"
    assert alias_args.controller == policy_render.SCRIPTED_REFERENCE_CONTROLLER
    assert alias_args.model_run_name == "ppo_line_smoke"
    assert alias_args.run_name == "scripted_line"
    assert alias_args.output_dir is None
    assert args.camera_distance == PARSER_CAMERA_DISTANCE
    assert args.camera_yaw == PARSER_CAMERA_YAW
    assert args.camera_pitch == PARSER_CAMERA_PITCH
    assert args.output_dir is None
    assert args.model_path.as_posix().endswith("storage/runs/direct_ppo_hover_seed0/training/models/direct_ppo_hover_seed0.zip")


def test_prepare_task_for_rollout_length_extends_short_reference() -> None:
    """Verify short tasks are densified so requested rollout lengths are reachable."""
    config = experiments_config.load_experiment_config("configs/smoke/trajectory_validation.yaml")
    line_task = dict(config["tasks"][2])

    prepared_task, reference_samples, warnings = policy_render._prepare_task_for_rollout_length(  # noqa: SLF001
        task=line_task,
        requested_max_steps=RENDER_MAX_STEPS,
    )

    assert prepared_task["shape"] == "line"
    assert reference_samples >= RENDER_REQUIRED_REFERENCE_SAMPLES
    assert warnings


def test_prepare_polyline_task_for_rollout_length_falls_back_to_duration_extension() -> None:
    """Verify polyline showcase tasks can be lengthened without invalid corner acceleration."""
    config = experiments_config.load_experiment_config("configs/smoke/trajectory_validation.yaml")
    polyline_task = dict(config["tasks"][4])

    prepared_task, reference_samples, warnings = policy_render._prepare_task_for_rollout_length(  # noqa: SLF001
        task=polyline_task,
        requested_max_steps=POLYLINE_RENDER_MAX_STEPS,
    )

    assert prepared_task["shape"] == "polyline"
    assert prepared_task["duration_sec"] > polyline_task["duration_sec"]
    assert prepared_task["sample_rate_hz"] == polyline_task["sample_rate_hz"]
    assert reference_samples >= POLYLINE_REQUIRED_REFERENCE_SAMPLES
    assert warnings
    assert "duration_sec" in warnings[0]


def test_policy_render_missing_model_path_raises_clear_error(tmp_path: Path) -> None:
    """Verify missing trained model paths fail with a useful training command hint."""
    missing_model = tmp_path / "missing_model.zip"

    with pytest.raises(FileNotFoundError, match="cli_train_tracking"):
        policy_render.run_trained_policy_render_from_paths(
            model_path=missing_model,
            config_path=Path("configs/training/ppo_tracking.yaml"),
            output_dir=tmp_path,
            max_steps=4,
            seed=0,
        )


def test_policy_render_manifest_includes_rollout_summary_fields() -> None:
    """Verify manifest payload includes requested fields for rollout explainability."""
    settings = policy_render.PolicyRenderSettings()
    payload = policy_render._build_manifest(  # noqa: SLF001
        settings=settings,
        evaluation_run_name=policy_render.DEFAULT_EVALUATION_RUN_NAME,
        mode=policy_render.TRAINED_POLICY_MODE,
        model_path=Path("storage/runs/direct_ppo_hover_seed0/training/models/direct_ppo_hover_seed0.zip"),
        configured_model_path=Path("storage/runs/direct_ppo_hover_seed0/training/models/direct_ppo_hover_seed0.zip"),
        training_task_shape="line",
        evaluation_task_shape="line",
        gif_path=Path("storage/runs/eval_direct_ppo_hover_seed0_on_hover/evaluations/policy_render/renders/trained_policy_rollout.gif"),
        trace_path=Path("storage/runs/eval_direct_ppo_hover_seed0_on_hover/evaluations/policy_render/traces/trained_policy_rollout_trace.jsonl"),
        plot_paths={
            "trajectory_xy": "storage/runs/eval_direct_ppo_hover_seed0_on_hover/evaluations/policy_render/plots/trajectory_xy.png",
            "position_error": "storage/runs/eval_direct_ppo_hover_seed0_on_hover/evaluations/policy_render/plots/position_error.png",
        },
        task_shape="line",
        task_source="config",
        task_index=LINE_TASK_INDEX,
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

    assert payload["mode"] == policy_render.TRAINED_POLICY_MODE
    assert payload["run_type"] == "evaluation"
    assert payload["evaluation_run_name"] == policy_render.DEFAULT_EVALUATION_RUN_NAME
    assert payload["controller_type"] == policy_render.PPO_CONTROLLER
    assert payload["baseline_type"] is None
    assert payload["model_run_name"] is None
    assert payload["model_path"] is not None
    assert payload["training_task_shape"] == "line"
    assert payload["evaluation_task_shape"] == "line"
    assert payload["render_task_shape"] == "line"
    assert payload["requested_max_steps"] == RENDER_MAX_STEPS
    assert payload["render_task_override_used"] is False
    assert payload["render_task_shape_requested"] is None
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
    assert payload["trace_path"].endswith("traces/trained_policy_rollout_trace.jsonl")
    assert payload["plot_paths"]["trajectory_xy"].endswith("trajectory_xy.png")
    assert payload["reference_path_overlay_enabled"] is True
    assert payload["waypoint_markers_enabled"] is True
    assert payload["active_target_marker_enabled"] is True
    assert payload["actual_path_trail_enabled"] is True
    assert payload["overlay_visual_roles"]["reference_path"]["color"] == "blue"
    assert payload["overlay_visual_roles"]["reference_waypoints"]["color"] == "yellow"
    assert payload["overlay_visual_roles"]["active_target"]["color"] == "green"
    assert payload["overlay_visual_roles"]["actual_path"]["color"] == "red"
    assert payload["overlay_geometry_mode"] == "pybullet_visual_only_no_collision"
    assert payload["trace_path"] in payload["output_files"]


def test_policy_render_review_artifact_dirs_use_run_subdirectories(tmp_path: Path) -> None:
    """Verify traces and plots are placed under the trained-policy render run root."""
    traces_dir, plots_dir = policy_render._review_artifact_dirs(tmp_path)  # noqa: SLF001

    assert traces_dir == tmp_path / "traces"
    assert plots_dir == tmp_path / "plots"


def test_policy_render_waypoint_positions_use_task_corners() -> None:
    """Verify waypoint overlay markers are derived from task geometry."""
    reference_positions = np.asarray(
        [[0.0, 0.0, 1.0], [0.5, 0.0, 1.0], [1.0, 0.0, 1.0]],
        dtype=float,
    )

    waypoints = policy_render._reference_waypoint_positions(  # noqa: SLF001
        task={"shape": "line", "start": [0.0, 0.0, 1.0], "end": [1.0, 0.0, 1.0]},
        reference_positions=reference_positions,
    )

    assert waypoints.tolist() == [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]


def test_policy_render_quaternion_from_z_axis_handles_axis_aligned_segments() -> None:
    """Verify overlay cylinder orientation helper returns normalized quaternions."""
    z_quaternion = policy_render._quaternion_from_z_axis(np.array([0.0, 0.0, 1.0]))  # noqa: SLF001
    x_quaternion = policy_render._quaternion_from_z_axis(np.array([1.0, 0.0, 0.0]))  # noqa: SLF001

    assert z_quaternion == [0.0, 0.0, 0.0, 1.0]
    assert sum(value * value for value in x_quaternion) == pytest.approx(1.0)


def test_policy_render_select_task_supports_showcase_shapes() -> None:
    """Verify configured trained-policy render tasks include the reviewer showcase shapes."""
    selected_shapes: list[str] = []
    selected_indices: list[int] = []
    for shape in SUPPORTED_RENDER_SHAPES:
        task, task_source, task_index, warnings = policy_render._select_task(  # noqa: SLF001
            task_config_path=Path("configs/smoke/trajectory_validation.yaml"),
            default_task_index=0,
            render_task_shape=shape,
        )
        selected_shapes.append(str(task["shape"]))
        selected_indices.append(task_index)
        assert task_source == "render_override"
        assert warnings

    assert tuple(selected_shapes) == SUPPORTED_RENDER_SHAPES
    assert len(set(selected_indices)) == len(SUPPORTED_RENDER_SHAPES)


def test_policy_render_manifest_marks_render_task_override() -> None:
    """Verify manifest payload explicitly reports render task shape overrides."""
    settings = policy_render.PolicyRenderSettings(render_task_shape="line")

    payload = policy_render._build_manifest(  # noqa: SLF001
        settings=settings,
        evaluation_run_name=policy_render.DEFAULT_EVALUATION_RUN_NAME,
        mode=policy_render.TRAINED_POLICY_MODE,
        model_path=Path("storage/runs/direct_ppo_hover_seed0/training/models/direct_ppo_hover_seed0.zip"),
        configured_model_path=Path("storage/runs/direct_ppo_hover_seed0/training/models/direct_ppo_hover_seed0.zip"),
        training_task_shape=None,
        evaluation_task_shape="line",
        gif_path=Path("storage/runs/eval_direct_ppo_hover_seed0_on_hover/evaluations/policy_render/renders/trained_policy_rollout.gif"),
        trace_path=Path("storage/runs/eval_direct_ppo_hover_seed0_on_hover/evaluations/policy_render/traces/trained_policy_rollout_trace.jsonl"),
        plot_paths={},
        task_shape="line",
        task_source="render_override",
        task_index=LINE_TASK_INDEX,
        requested_max_steps=4,
        actual_steps=4,
        termination_reason="tracking_max_steps_reached",
        position_bounds={"min": [0.0, 0.0, 1.0], "max": [1.0, 0.0, 1.0]},
        reference_position_bounds={"min": [0.0, 0.0, 1.0], "max": [1.0, 0.0, 1.0]},
        true_simulator_rendering=True,
        policy_predict_used=True,
        metrics={"mode": "trained_policy_render", "steps": 4},
        warnings=(),
    )

    assert payload["task_shape"] == "line"
    assert payload["task_source"] == "render_override"
    assert payload["task_index"] == LINE_TASK_INDEX
    assert payload["render_task_override_used"] is True
    assert payload["render_task_shape_requested"] == "line"


def test_policy_render_manifest_marks_scripted_reference_baseline() -> None:
    """Verify scripted baseline manifests cannot be mistaken for PPO renders."""
    settings = policy_render.PolicyRenderSettings(
        controller=policy_render.SCRIPTED_REFERENCE_CONTROLLER,
        run_name="scripted_reference_render_line",
        render_task_shape="line",
    )

    payload = policy_render._build_manifest(  # noqa: SLF001
        settings=settings,
        evaluation_run_name="eval_scripted_reference_on_line",
        mode=policy_render.SCRIPTED_REFERENCE_MODE,
        model_path=None,
        configured_model_path=Path("storage/runs/direct_ppo_hover_seed0/training/models/direct_ppo_hover_seed0.zip"),
        training_task_shape=None,
        evaluation_task_shape="line",
        gif_path=Path("storage/runs/eval_scripted_reference_on_line/evaluations/policy_render/renders/trained_policy_rollout.gif"),
        trace_path=Path("storage/runs/eval_scripted_reference_on_line/evaluations/policy_render/traces/trained_policy_rollout_trace.jsonl"),
        plot_paths={"trajectory_xy": "storage/runs/eval_scripted_reference_on_line/evaluations/policy_render/plots/trajectory_xy.png"},
        task_shape="line",
        task_source="render_override",
        task_index=LINE_TASK_INDEX,
        requested_max_steps=4,
        actual_steps=4,
        termination_reason="tracking_max_steps_reached",
        position_bounds={"min": [0.0, 0.0, 1.0], "max": [1.0, 0.0, 1.0]},
        reference_position_bounds={"min": [0.0, 0.0, 1.0], "max": [1.0, 0.0, 1.0]},
        true_simulator_rendering=True,
        policy_predict_used=False,
        metrics={"mode": "scripted_reference_baseline", "steps": 4},
        warnings=(),
        output_dir=Path("storage/runs/eval_scripted_reference_on_line/evaluations/policy_render"),
    )

    assert payload["mode"] == policy_render.SCRIPTED_REFERENCE_MODE
    assert payload["controller_type"] == policy_render.SCRIPTED_REFERENCE_CONTROLLER
    assert payload["baseline_type"] == policy_render.SCRIPTED_REFERENCE_BASELINE_TYPE
    assert payload["policy_predict_used"] is False
    assert payload["model_path"] is None
    assert payload["run_name"] == "scripted_reference_render_line"
    assert payload["trace_path"].endswith("eval_scripted_reference_on_line/evaluations/policy_render/traces/trained_policy_rollout_trace.jsonl")


def test_policy_render_resolves_model_run_name_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify model-run-name resolves to the selected training run model artifact."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    settings = policy_render.PolicyRenderSettings(model_run_name="ppo_line_smoke")

    model_path = policy_render._resolve_model_path(settings)  # noqa: SLF001

    assert model_path == tmp_path / "runs" / "ppo_line_smoke" / "training" / "models" / "ppo_line_smoke.zip"


def test_policy_render_loads_training_task_shape_from_model_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify render metadata can report the task shape used during training."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    metrics_dir = tmp_path / "runs" / "ppo_line_smoke" / "training" / "metrics"
    metrics_dir.mkdir(parents=True)
    metrics_path = metrics_dir / "ppo_line_smoke_metrics.json"
    metrics_path.write_text(json.dumps({"training_task_shape": "line"}), encoding="utf-8")

    metadata, warnings = policy_render._load_training_metadata("ppo_line_smoke")  # noqa: SLF001

    assert warnings == ()
    assert policy_render._training_task_shape(metadata) == "line"  # noqa: SLF001


def test_policy_render_resolves_run_name_output_dir() -> None:
    """Verify run names place artifacts under storage/runs evaluations without overriding explicit output dirs."""
    run_dir = policy_render._resolve_output_dir(None, "scripted_reference_render_line")  # noqa: SLF001
    explicit_dir = Path("storage/runs/custom_render/evaluations/policy_render")

    assert run_dir.as_posix().endswith("storage/runs/scripted_reference_render_line/evaluations/policy_render")
    assert policy_render._resolve_output_dir(explicit_dir, "ignored") == explicit_dir.resolve(strict=False)  # noqa: SLF001


def test_policy_render_controller_action_supports_ppo_and_scripted_reference() -> None:
    """Verify controller dispatch keeps PPO prediction separate from scripted reference actions."""

    class FakeModel:
        def predict(self, observation: np.ndarray, deterministic: bool) -> tuple[np.ndarray, None]:
            assert deterministic is True
            assert observation.shape == (1,)
            return np.asarray([[0.1, 0.2, 1.0]], dtype=float), None

    class FakeReference:
        positions = np.asarray([[0.0, 0.0, 1.0], [0.5, 0.0, 1.0]], dtype=float)

    class FakeActionSpace:
        shape = (1, 3)

    class FakeEnv:
        reference = FakeReference()
        action_space = FakeActionSpace()

    ppo_action, ppo_used = policy_render._controller_action(  # noqa: SLF001
        model=FakeModel(),
        observation=np.asarray([1.0], dtype=float),
        tracking_env=FakeEnv(),
        controller=policy_render.PPO_CONTROLLER,
        step_index=0,
    )
    scripted_action, scripted_used = policy_render._controller_action(  # noqa: SLF001
        model=None,
        observation=np.asarray([1.0], dtype=float),
        tracking_env=FakeEnv(),
        controller=policy_render.SCRIPTED_REFERENCE_CONTROLLER,
        step_index=SCRIPTED_ACTION_STEP_INDEX,
    )

    assert ppo_used is True
    assert ppo_action.tolist() == [[0.1, 0.2, 1.0]]
    assert scripted_used is False
    assert scripted_action.tolist() == [[0.5, 0.0, 1.0]]


def test_policy_render_artifact_dirs_preserve_explicit_output_override(tmp_path: Path) -> None:
    """Verify storage/results-style output overrides preserve direct render placement."""
    explicit_output_dir = tmp_path / "storage" / "results" / "trained_policy_render"

    renders_dir, manifests_dir = policy_render._artifact_dirs(explicit_output_dir)  # noqa: SLF001

    assert renders_dir == explicit_output_dir
    assert manifests_dir == explicit_output_dir


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

    result = policy_render._write_manifest(manifest_path, payload)  # noqa: SLF001
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result == payload
    assert loaded == payload


def test_cli_render_policy_help_works() -> None:
    """Verify the trained-policy render CLI help path works without running rendering."""
    completed = subprocess.run(
        [sys.executable, "-m", "src.experiments.cli.experiments_cli_render_policy", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--model-path" in completed.stdout
    assert "--model-run-name" in completed.stdout
    assert "--task-shape" in completed.stdout
    assert "--render-task-shape" in completed.stdout
    assert "--controller" in completed.stdout
    assert "--run-name" in completed.stdout
    assert "--camera-distance" in completed.stdout
    assert "--camera-yaw" in completed.stdout
    assert "--camera-pitch" in completed.stdout


class _EnvWrapper:
    def __init__(self, env: object) -> None:
        self.env = env


class _UnwrappedWrapper:
    def __init__(self, unwrapped: object) -> None:
        self.unwrapped = unwrapped


def test_resolve_tracking_env_for_rendering_accepts_direct_tracking_env() -> None:
    """Verify direct TrajectoryTrackingEnv instances resolve without wrapping."""
    tracking_env = object.__new__(envs.tracking_env.TrajectoryTrackingEnv)

    resolved = policy_render.resolve_tracking_env_for_rendering(tracking_env)

    assert resolved is tracking_env


def test_resolve_tracking_env_for_rendering_unwraps_env_chain() -> None:
    """Verify Gymnasium-style .env wrappers resolve to the tracking base env."""
    tracking_env = object.__new__(envs.tracking_env.TrajectoryTrackingEnv)
    wrapped = _EnvWrapper(_EnvWrapper(tracking_env))

    resolved = policy_render.resolve_tracking_env_for_rendering(wrapped)

    assert resolved is tracking_env


def test_resolve_tracking_env_for_rendering_unwraps_unwrapped_chain() -> None:
    """Verify .unwrapped chains resolve to the tracking base env."""
    tracking_env = object.__new__(envs.tracking_env.TrajectoryTrackingEnv)
    wrapped = _UnwrappedWrapper(_UnwrappedWrapper(tracking_env))

    resolved = policy_render.resolve_tracking_env_for_rendering(wrapped)

    assert resolved is tracking_env


def test_resolve_tracking_env_for_rendering_fails_for_incompatible_object() -> None:
    """Verify incompatible objects raise a clear rendering resolution error."""
    with pytest.raises(RuntimeError, match="could not resolve TrajectoryTrackingEnv"):
        policy_render.resolve_tracking_env_for_rendering(object())


def test_resolve_tracking_env_for_rendering_does_not_loop_forever() -> None:
    """Verify cyclic wrappers terminate and fail clearly instead of looping forever."""

    class _Cyclic:
        def __init__(self) -> None:
            self.env = self
            self.unwrapped = self

    with pytest.raises(RuntimeError, match="could not resolve TrajectoryTrackingEnv"):
        policy_render.resolve_tracking_env_for_rendering(_Cyclic())
