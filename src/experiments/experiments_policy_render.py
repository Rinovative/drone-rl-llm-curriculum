"""
===============================================================================
experiments_policy_render.py
===============================================================================
Render a trained PPO policy rollout on TrajectoryTrackingEnv with external camera capture.

Responsibilities:
  - Load PPO smoke settings and a persisted Stable-Baselines3 PPO model
  - Run a bounded deterministic policy rollout on TrajectoryTrackingEnv
  - Capture true simulator external-camera frames and encode a GIF artifact
  - Write rollout metrics and a JSON manifest under approved storage paths

Design principles:
  - Keep execution headless, deterministic, bounded, and reviewer-friendly
  - Import heavyweight RL/render dependencies lazily inside runtime functions
  - Fail clearly when trained-policy prerequisites are unavailable

Boundaries:
  - PPO training belongs in experiments_ppo_tracking.py and related CLI modules
  - Environment/task internals belong in envs and validation packages
===============================================================================

"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src import envs, experiments, utils

DEFAULT_PPO_CONFIG_PATH = Path("configs/smoke/ppo_tracking_smoke.yaml")
DEFAULT_MODEL_PATH = Path("storage/runs/ppo_tracking_smoke/models/ppo_tracking_smoke.zip")
DEFAULT_OUTPUT_DIR = Path("storage/runs/trained_policy_render")
DEFAULT_MAX_STEPS = 60
DEFAULT_SEED = 0
DEFAULT_CAMERA_MODE = "follow_external"
SUPPORTED_CAMERA_MODES = ("follow_external", "fixed_external")
DEFAULT_GIF_FILENAME = "trained_policy_rollout.gif"
DEFAULT_MANIFEST_FILENAME = "trained_policy_render_manifest.json"


def default_model_path() -> Path:
    """Return the default trained PPO model path for reviewer/demo rendering."""
    return utils.artifacts.get_models_dir("ppo_tracking_smoke") / "ppo_tracking_smoke.zip"


def default_output_dir() -> Path:
    """Return the default trained-policy render run directory."""
    return utils.artifacts.get_run_dir("trained_policy_render")


DEFAULT_IMAGE_WIDTH = 480
DEFAULT_IMAGE_HEIGHT = 360
DEFAULT_FRAME_INTERVAL = 1
EARLY_TRUNCATION_WARNING_FRACTION = 0.5
DEFAULT_CAMERA_DISTANCE_M = 1.15
DEFAULT_CAMERA_YAW_DEG = 40.0
DEFAULT_CAMERA_PITCH_DEG = -23.0
_DEFAULT_CAMERA_FOV_DEG = 44.0
_DEFAULT_FAR_CLIP_M = 25.0
_DEFAULT_NEAR_CLIP_M = 0.05


@dataclass(frozen=True)
class PolicyRenderSettings:
    """
    Settings for trained PPO policy rollout rendering.

    Parameters
    ----------
    model_path
        Path to a saved Stable-Baselines3 PPO ``.zip`` model.
    config_path
        PPO tracking smoke YAML path used to resolve task selection defaults.
    task_index
        Optional task index override. Uses config default when omitted.
    render_task_shape
        Optional render-only task-shape override selected from the task config.
    output_dir
        Directory where GIF and manifest artifacts are written.
    max_steps
        Maximum rollout steps to execute.
    seed
        Deterministic seed forwarded to environment reset.
    camera_mode
        External camera mode. Supported values are ``follow_external`` and ``fixed_external``.
    camera_distance
        External camera distance from the target, in meters.
    camera_yaw
        External camera yaw in degrees.
    camera_pitch
        External camera pitch in degrees.
    gif_filename
        GIF filename written under ``output_dir``.
    manifest_filename
        JSON manifest filename written under ``output_dir``.
    frame_interval
        Number of environment steps between camera captures.
    image_width
        Captured frame width in pixels.
    image_height
        Captured frame height in pixels.

    """

    model_path: Path = field(default_factory=default_model_path)
    config_path: Path = DEFAULT_PPO_CONFIG_PATH
    task_index: int | None = None
    render_task_shape: str | None = None
    output_dir: Path | None = None
    max_steps: int = DEFAULT_MAX_STEPS
    seed: int | None = DEFAULT_SEED
    camera_mode: str = DEFAULT_CAMERA_MODE
    camera_distance: float = DEFAULT_CAMERA_DISTANCE_M
    camera_yaw: float = DEFAULT_CAMERA_YAW_DEG
    camera_pitch: float = DEFAULT_CAMERA_PITCH_DEG
    gif_filename: str = DEFAULT_GIF_FILENAME
    manifest_filename: str = DEFAULT_MANIFEST_FILENAME
    frame_interval: int = DEFAULT_FRAME_INTERVAL
    image_width: int = DEFAULT_IMAGE_WIDTH
    image_height: int = DEFAULT_IMAGE_HEIGHT

    def __post_init__(self) -> None:
        """Validate policy-render settings."""
        if self.task_index is not None and self.task_index < 0:
            message = "task_index must be nonnegative when provided"
            raise ValueError(message)
        if self.max_steps <= 0:
            message = "max_steps must be positive"
            raise ValueError(message)
        if self.seed is not None and self.seed < 0:
            message = "seed must be nonnegative"
            raise ValueError(message)
        if self.frame_interval <= 0:
            message = "frame_interval must be positive"
            raise ValueError(message)
        if self.image_width <= 0 or self.image_height <= 0:
            message = "image dimensions must be positive"
            raise ValueError(message)
        if self.camera_mode not in SUPPORTED_CAMERA_MODES:
            message = f"camera_mode must be one of: {', '.join(SUPPORTED_CAMERA_MODES)}"
            raise ValueError(message)
        if not np.isfinite(self.camera_distance) or self.camera_distance <= 0.0:
            message = "camera_distance must be finite and positive"
            raise ValueError(message)
        if not np.isfinite(self.camera_yaw):
            message = "camera_yaw must be finite"
            raise ValueError(message)
        if not np.isfinite(self.camera_pitch):
            message = "camera_pitch must be finite"
            raise ValueError(message)
        if not self.model_path.name.endswith(".zip"):
            message = "model_path must point to a .zip file"
            raise ValueError(message)
        if not self.gif_filename.endswith(".gif"):
            message = "gif_filename must end with .gif"
            raise ValueError(message)
        if not self.manifest_filename.endswith(".json"):
            message = "manifest_filename must end with .json"
            raise ValueError(message)


@dataclass(frozen=True)
class PolicyRenderResult:
    """
    Summary returned by a trained-policy render run.

    Parameters
    ----------
    gif_path
        Path to the written policy-rollout GIF.
    manifest_path
        Path to the written JSON manifest.
    metrics
        JSON-serializable rollout metrics payload.
    warnings
        Nonfatal warning strings generated during the run.

    """

    gif_path: str
    manifest_path: str
    metrics: dict[str, Any]
    warnings: tuple[str, ...] = ()


def run_trained_policy_render(settings: PolicyRenderSettings | None = None) -> PolicyRenderResult:
    """
    Load a trained PPO model, run a bounded rollout, and write GIF/manifest artifacts.

    Parameters
    ----------
    settings
        Optional policy-render settings. Defaults are used when omitted.

    Returns
    -------
    PolicyRenderResult
        Paths and metrics proving a trained policy was rendered with simulator frames.

    Raises
    ------
    FileNotFoundError
        If the configured model path does not exist.
    RuntimeError
        If PPO loading, environment stepping, or simulator camera rendering fails.

    """
    active_settings = settings or PolicyRenderSettings()
    model_path = active_settings.model_path.expanduser().resolve(strict=False)
    if not model_path.exists():
        message = (
            "trained PPO model was not found at "
            f"{model_path}. Create it with: "
            "python -m src.experiments.cli_train_tracking --config configs/smoke/ppo_tracking_smoke.yaml"
        )
        raise FileNotFoundError(message)

    ppo_settings = experiments.ppo_tracking.load_ppo_tracking_settings(active_settings.config_path)
    seed = ppo_settings.seed if active_settings.seed is None else active_settings.seed
    task, task_source, selected_task_index, selection_warnings = _select_task(
        task_config_path=ppo_settings.task_config_path,
        default_task_index=ppo_settings.task_index if active_settings.task_index is None else active_settings.task_index,
        render_task_shape=active_settings.render_task_shape,
    )
    prepared_task, reference_sample_count, preparation_warnings = _prepare_task_for_rollout_length(
        task=task,
        requested_max_steps=active_settings.max_steps,
    )

    warnings: list[str] = [*selection_warnings, *preparation_warnings]

    output_dir = _resolve_directory(active_settings.output_dir, default_output_dir())
    renders_dir, manifests_dir = _artifact_dirs(output_dir)
    renders_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    gif_path = renders_dir / active_settings.gif_filename
    manifest_path = manifests_dir / active_settings.manifest_filename

    try:
        from stable_baselines3 import PPO  # noqa: PLC0415
    except ImportError as exc:
        message = f"stable_baselines3 is required for trained-policy rendering: {exc}"
        raise RuntimeError(message) from exc

    model = PPO.load(str(model_path), device="cpu")
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(
        prepared_task,
        gui=False,
        record=False,
        max_steps=active_settings.max_steps,
    )

    try:
        rollout_payload = _run_policy_rollout(
            model=model,
            tracking_env=tracking_env,
            settings=active_settings,
            seed=seed,
            task_shape=str(prepared_task.get("shape", "unknown")),
        )
    finally:
        tracking_env.close()

    _write_gif(rollout_payload["frames"], gif_path, active_settings.frame_interval)
    actual_steps = len(rollout_payload["rewards"])
    position_bounds = _position_bounds(rollout_payload["current_positions"])
    reference_position_bounds = _position_bounds(rollout_payload["reference_positions"])
    final_info = dict(rollout_payload.get("final_info", {}))
    termination_reason = str(
        final_info.get(
            "termination_reason",
            _termination_reason(
                terminated=rollout_payload["terminated"],
                truncated=rollout_payload["truncated"],
                actual_steps=actual_steps,
                requested_max_steps=active_settings.max_steps,
                reference_sample_count=reference_sample_count,
            ),
        )
    )
    warnings.extend(
        _rollout_warnings(
            actual_steps=actual_steps,
            requested_max_steps=active_settings.max_steps,
            terminated=bool(rollout_payload["terminated"]),
            truncated=bool(rollout_payload["truncated"]),
            termination_reason=termination_reason,
        )
    )

    metrics = _build_metrics(
        rewards=rollout_payload["rewards"],
        position_errors=rollout_payload["position_errors"],
        task_shape=str(prepared_task.get("shape", "unknown")),
        model_path=model_path,
        terminated=rollout_payload["terminated"],
        truncated=rollout_payload["truncated"],
        warnings=tuple(warnings),
    )
    manifest = _build_manifest(
        settings=active_settings,
        model_path=model_path,
        gif_path=gif_path,
        task_shape=str(prepared_task.get("shape", "unknown")),
        task_source=task_source,
        task_index=selected_task_index,
        requested_max_steps=active_settings.max_steps,
        actual_steps=actual_steps,
        termination_reason=termination_reason,
        position_bounds=position_bounds,
        reference_position_bounds=reference_position_bounds,
        true_simulator_rendering=True,
        policy_predict_used=bool(rollout_payload["policy_predict_used"]),
        metrics=metrics,
        warnings=tuple(warnings),
        final_info=final_info,
        final_action=rollout_payload.get("final_action"),
    )
    _write_manifest(manifest_path, manifest)

    return PolicyRenderResult(
        gif_path=str(gif_path),
        manifest_path=str(manifest_path),
        metrics=metrics,
        warnings=tuple(warnings),
    )


def run_trained_policy_render_from_paths(
    model_path: Path,
    config_path: Path = DEFAULT_PPO_CONFIG_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int | None = DEFAULT_SEED,
    camera_mode: str = DEFAULT_CAMERA_MODE,
    task_index: int | None = None,
    render_task_shape: str | None = None,
    camera_distance: float = DEFAULT_CAMERA_DISTANCE_M,
    camera_yaw: float = DEFAULT_CAMERA_YAW_DEG,
    camera_pitch: float = DEFAULT_CAMERA_PITCH_DEG,
) -> PolicyRenderResult:
    """
    Build settings from explicit paths and run trained-policy rendering.

    Parameters
    ----------
    model_path
        Path to a saved Stable-Baselines3 PPO model.
    config_path
        PPO tracking smoke YAML path used to resolve task defaults.
    output_dir
        Directory where GIF and manifest artifacts are written.
    max_steps
        Maximum rollout steps.
    seed
        Deterministic reset seed.
    camera_mode
        External camera mode.
    task_index
        Optional task index override.
    render_task_shape
        Optional render-only task-shape override selected from the task config.
    camera_distance
        External camera distance from the target, in meters.
    camera_yaw
        External camera yaw in degrees.
    camera_pitch
        External camera pitch in degrees.

    Returns
    -------
    PolicyRenderResult
        Render artifact paths and rollout metrics.

    """
    return run_trained_policy_render(
        PolicyRenderSettings(
            model_path=model_path,
            config_path=config_path,
            task_index=task_index,
            render_task_shape=render_task_shape,
            output_dir=output_dir,
            max_steps=max_steps,
            seed=seed,
            camera_mode=camera_mode,
            camera_distance=camera_distance,
            camera_yaw=camera_yaw,
            camera_pitch=camera_pitch,
        )
    )


def _artifact_dirs(output_dir: Path) -> tuple[Path, Path]:
    """Return render and manifest directories for a run-root or legacy output override."""
    if "results" in output_dir.parts:
        return output_dir, output_dir
    return output_dir / "renders", output_dir / "manifests"


def _run_policy_rollout(
    model: Any,
    tracking_env: Any,
    settings: PolicyRenderSettings,
    seed: int,
    task_shape: str,
) -> dict[str, Any]:
    """Step TrajectoryTrackingEnv with deterministic PPO actions and capture camera frames."""
    observation, info = tracking_env.reset(seed=seed)

    rewards: list[float] = []
    position_errors: list[float] = []
    current_positions: list[np.ndarray] = []
    reference_positions: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    policy_predict_used = False
    final_info: dict[str, Any] = dict(info)
    final_action: np.ndarray | None = None

    initial_position = np.asarray(info.get("current_position"), dtype=float)
    _capture_external_frame(frames=frames, tracking_env=tracking_env, settings=settings, position=initial_position)

    terminated = False
    truncated = False
    for step_index in range(settings.max_steps):
        action, _ = model.predict(observation, deterministic=True)
        final_action = np.asarray(action, dtype=float)
        policy_predict_used = True
        observation, reward, terminated, truncated, info = tracking_env.step(action)
        final_info = dict(info)

        current_position = np.asarray(info["current_position"], dtype=float)
        reference_position = np.asarray(info["reference_position"], dtype=float)

        current_positions.append(current_position)
        reference_positions.append(reference_position)
        rewards.append(float(reward))
        position_errors.append(float(info["position_error_m"]))

        if step_index % settings.frame_interval == 0 or terminated or truncated:
            _capture_external_frame(frames=frames, tracking_env=tracking_env, settings=settings, position=current_position)

        if terminated or truncated:
            break

    if not frames:
        message = f"no simulator frames were captured for task '{task_shape}'"
        raise RuntimeError(message)

    return {
        "frames": frames,
        "rewards": rewards,
        "position_errors": position_errors,
        "current_positions": current_positions,
        "reference_positions": reference_positions,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "policy_predict_used": policy_predict_used,
        "final_info": final_info,
        "final_action": final_action,
    }


def _capture_external_frame(frames: list[np.ndarray], tracking_env: Any, settings: PolicyRenderSettings, position: np.ndarray) -> None:
    """Capture an RGB frame from an external third-person camera in the simulator."""
    try:
        import pybullet as p  # noqa: PLC0415
    except ImportError as exc:
        message = f"pybullet is required for external camera rendering: {exc}"
        raise RuntimeError(message) from exc

    base_env = getattr(tracking_env, "base_env", None)
    client_id = getattr(base_env, "CLIENT", None)
    if base_env is None or client_id is None:
        message = "TrajectoryTrackingEnv does not expose the required simulator client for camera capture"
        raise RuntimeError(message)

    target = _camera_target(camera_mode=settings.camera_mode, position=position)
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target.tolist(),
        distance=settings.camera_distance,
        yaw=settings.camera_yaw,
        pitch=settings.camera_pitch,
        roll=0.0,
        upAxisIndex=2,
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=_DEFAULT_CAMERA_FOV_DEG,
        aspect=settings.image_width / settings.image_height,
        nearVal=_DEFAULT_NEAR_CLIP_M,
        farVal=_DEFAULT_FAR_CLIP_M,
    )
    _, _, rgb, _, _ = p.getCameraImage(
        width=settings.image_width,
        height=settings.image_height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        flags=p.ER_NO_SEGMENTATION_MASK,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client_id,
    )
    frame = np.reshape(np.asarray(rgb, dtype=np.uint8), (settings.image_height, settings.image_width, 4))[..., :3]
    frames.append(frame)


def _camera_target(camera_mode: str, position: np.ndarray) -> np.ndarray:
    """Return the camera target point for a supported external camera mode."""
    flight_center = np.array([0.0, 0.0, 1.0], dtype=float)
    if camera_mode == "fixed_external":
        return flight_center
    return (0.2 * flight_center) + (0.8 * position)


def _write_gif(frames: list[np.ndarray], path: Path, frame_interval: int) -> None:
    """Encode simulator frames as a GIF artifact."""
    if not frames:
        message = "cannot write trained-policy GIF without frames"
        raise RuntimeError(message)

    try:
        import imageio.v2 as imageio  # noqa: PLC0415
    except ImportError as exc:
        message = f"imageio is required to encode rollout GIF artifacts: {exc}"
        raise RuntimeError(message) from exc

    frame_duration = max(frame_interval / 30.0, 0.04)
    frame_payload: Any = frames
    imageio.mimsave(path, frame_payload, duration=frame_duration)


def _build_metrics(
    rewards: list[float],
    position_errors: list[float],
    task_shape: str,
    model_path: Path,
    terminated: bool,
    truncated: bool,
    warnings: tuple[str, ...],
) -> dict[str, Any]:
    """Build required trained-policy rollout metrics."""
    if not rewards or not position_errors:
        message = "trained-policy rollout produced no step metrics"
        raise RuntimeError(message)

    return {
        "mode": "trained_policy_render",
        "model_path": str(model_path),
        "task_shape": task_shape,
        "steps": len(rewards),
        "mean_reward": float(np.mean(rewards)),
        "final_reward": float(rewards[-1]),
        "mean_position_error_m": float(np.mean(position_errors)),
        "final_position_error_m": float(position_errors[-1]),
        "min_position_error_m": float(np.min(position_errors)),
        "max_position_error_m": float(np.max(position_errors)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "warnings": list(warnings),
    }


def _build_manifest(
    settings: PolicyRenderSettings,
    model_path: Path,
    gif_path: Path,
    task_shape: str,
    task_source: str,
    task_index: int,
    requested_max_steps: int,
    actual_steps: int,
    termination_reason: str,
    position_bounds: dict[str, list[float]],
    reference_position_bounds: dict[str, list[float]],
    true_simulator_rendering: bool,
    policy_predict_used: bool,
    metrics: dict[str, Any],
    warnings: tuple[str, ...],
    final_info: dict[str, Any] | None = None,
    final_action: Any | None = None,
) -> dict[str, Any]:
    """Build a trained-policy render manifest payload."""
    info = {} if final_info is None else final_info
    survived_fraction = _survived_fraction(actual_steps=actual_steps, requested_max_steps=requested_max_steps)
    return {
        "mode": "trained_policy_render",
        "render_mode": "simulator_external_camera_gif",
        "task_shape": task_shape,
        "task_source": task_source,
        "task_index": task_index,
        "requested_max_steps": requested_max_steps,
        "actual_steps": actual_steps,
        "survived_fraction": survived_fraction,
        "termination_reason": termination_reason,
        "base_terminated": bool(info.get("base_terminated", False)),
        "base_truncated": bool(info.get("base_truncated", False)),
        "base_info_keys": list(info.get("base_info_keys", [])),
        "base_reason_fields": dict(info.get("base_reason_fields", {})),
        "base_truncation_causes": list(info.get("base_truncation_causes", [])),
        "camera_mode": settings.camera_mode,
        "camera_distance": float(settings.camera_distance),
        "camera_yaw": float(settings.camera_yaw),
        "camera_pitch": float(settings.camera_pitch),
        "camera_settings": {
            "mode": settings.camera_mode,
            "distance_m": float(settings.camera_distance),
            "yaw_deg": float(settings.camera_yaw),
            "pitch_deg": float(settings.camera_pitch),
            "image_width": int(settings.image_width),
            "image_height": int(settings.image_height),
            "frame_interval": int(settings.frame_interval),
        },
        "position_bounds": position_bounds,
        "reference_position_bounds": reference_position_bounds,
        "z_span_m": _axis_span(position_bounds, axis=2),
        "xy_span_m": float(np.linalg.norm([_axis_span(position_bounds, axis=0), _axis_span(position_bounds, axis=1)])),
        "final_position": _array_to_jsonable(info.get("current_position", [])),
        "final_reference_position": _array_to_jsonable(info.get("reference_position", [])),
        "final_position_error_m": float(info.get("position_error_m", 0.0)),
        "final_attitude_rpy": _array_to_jsonable(info.get("roll_pitch_yaw", [])),
        "final_velocity": _array_to_jsonable(info.get("velocity", [])),
        "final_angular_velocity": _array_to_jsonable(info.get("angular_velocity", [])),
        "final_action": _array_to_jsonable(final_action if final_action is not None else []),
        "final_last_action": _array_to_jsonable(info.get("last_action", [])),
        "true_simulator_rendering": bool(true_simulator_rendering),
        "policy_predict_used": bool(policy_predict_used),
        "model_path": str(model_path),
        "gif_path": str(gif_path),
        "output_files": [str(gif_path)],
        "metrics": metrics,
        "warnings": list(warnings),
    }


def _rollout_warnings(
    actual_steps: int,
    requested_max_steps: int,
    terminated: bool,
    truncated: bool,
    termination_reason: str,
) -> list[str]:
    """Return manifest warnings for suspiciously short trained-policy rollouts."""
    if requested_max_steps <= 0:
        return []
    survived_fraction = _survived_fraction(actual_steps=actual_steps, requested_max_steps=requested_max_steps)
    if survived_fraction >= EARLY_TRUNCATION_WARNING_FRACTION:
        return []
    if not (terminated or truncated):
        return []
    return [f"trained policy rollout ended after {actual_steps}/{requested_max_steps} steps ({survived_fraction:.3f}); reason={termination_reason}"]


def _survived_fraction(actual_steps: int, requested_max_steps: int) -> float:
    """Return the fraction of requested rollout steps actually executed."""
    if requested_max_steps <= 0:
        return 0.0
    return float(actual_steps / requested_max_steps)


def _axis_span(bounds: dict[str, list[float]], axis: int) -> float:
    """Return the span of one bounded axis."""
    mins = bounds.get("min", [])
    maxes = bounds.get("max", [])
    if len(mins) <= axis or len(maxes) <= axis:
        return 0.0
    return float(maxes[axis] - mins[axis])


def _array_to_jsonable(value: Any) -> list[Any]:
    """Convert an array-like value to nested JSON-compatible lists."""
    return np.asarray(value).tolist()


def _write_manifest(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Write a manifest payload to JSON and return the same payload."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _resolve_directory(path: Path | None, default: Path) -> Path:
    """Resolve a configured directory or fallback default without strict existence checks."""
    directory = default if path is None else path
    return directory.expanduser().resolve(strict=False)


def _select_task(
    task_config_path: Path,
    default_task_index: int,
    render_task_shape: str | None,
) -> tuple[dict[str, Any], str, int, tuple[str, ...]]:
    """Load one task from config by index or render-task-shape override."""
    config = experiments.config.load_experiment_config(task_config_path)
    tasks = config.get("tasks")
    if not isinstance(tasks, list):
        message = "task config must contain a top-level tasks list"
        raise ValueError(message)  # noqa: TRY004 - public config contract reports config errors as ValueError.

    warnings: list[str] = []
    if render_task_shape is None:
        if default_task_index < 0 or default_task_index >= len(tasks):
            message = "task_index is outside the configured task list"
            raise ValueError(message)
        task = tasks[default_task_index]
        if not isinstance(task, dict):
            message = "selected task must be a mapping"
            raise ValueError(message)
        return dict(task), "config", default_task_index, tuple(warnings)

    requested_shape = render_task_shape.strip().lower()
    if not requested_shape:
        message = "render_task_shape must be non-empty when provided"
        raise ValueError(message)

    for index, candidate in enumerate(tasks):
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("shape", "")).lower() == requested_shape:
            warnings.append("render_task_shape override selected a render task from config")
            return dict(candidate), "render_override", index, tuple(warnings)

    available_shapes = sorted({str(task.get("shape")) for task in tasks if isinstance(task, dict) and task.get("shape") is not None})
    message = f"render_task_shape '{render_task_shape}' not found in task config; available: {', '.join(available_shapes)}"
    raise ValueError(message)


def _prepare_task_for_rollout_length(task: dict[str, Any], requested_max_steps: int) -> tuple[dict[str, Any], int, tuple[str, ...]]:
    """Extend short tasks by increasing sample rate so render max-steps remain meaningful."""
    reference = envs.task_adapter.make_task_reference(task)
    reference_samples = int(reference.positions.shape[0])
    required_samples = requested_max_steps + 1
    if reference_samples >= required_samples:
        return dict(task), reference_samples, ()

    shape = str(task.get("shape", "unknown"))
    duration_value = task.get("duration_sec")
    sample_rate_value = task.get("sample_rate_hz")
    if duration_value is None or sample_rate_value is None:
        warning = "selected task has too few samples for requested max_steps and cannot be extended because duration_sec/sample_rate_hz are missing"
        return dict(task), reference_samples, (warning,)

    duration_sec = float(duration_value)
    sample_rate_hz = float(sample_rate_value)
    if duration_sec <= 0.0 or sample_rate_hz <= 0.0:
        warning = "selected task has non-positive duration_sec or sample_rate_hz and cannot be safely extended"
        return dict(task), reference_samples, (warning,)

    required_sample_rate_hz = int(np.ceil(required_samples / duration_sec))
    if required_sample_rate_hz <= sample_rate_hz:
        required_sample_rate_hz = int(np.ceil(sample_rate_hz * required_samples / max(reference_samples, 1)))

    extended_task = dict(task)
    extended_task["sample_rate_hz"] = float(required_sample_rate_hz)
    extended_reference = envs.task_adapter.make_task_reference(extended_task)
    extended_samples = int(extended_reference.positions.shape[0])
    warning = (
        f"extended render task sample_rate_hz from {sample_rate_hz} to {required_sample_rate_hz} "
        f"for shape '{shape}' so rollout can reach requested max_steps"
    )
    if extended_samples < required_samples:
        warning = warning + "; rollout may still end early due to reference trajectory length"
    return extended_task, extended_samples, (warning,)


def _termination_reason(
    terminated: bool,
    truncated: bool,
    actual_steps: int,
    requested_max_steps: int,
    reference_sample_count: int,
) -> str:
    """Explain why the rollout ended for manifest review."""
    if truncated:
        return "truncated"
    if terminated and actual_steps >= reference_sample_count and actual_steps >= requested_max_steps:
        return "terminated_reference_complete_and_requested_max_steps"
    if terminated and actual_steps >= reference_sample_count:
        return "terminated_reference_complete"
    if terminated and actual_steps >= requested_max_steps:
        return "terminated_requested_max_steps"
    if terminated:
        return "terminated"
    if actual_steps >= requested_max_steps:
        return "requested_max_steps_exhausted_without_terminal"
    return "rollout_loop_ended_early"


def _position_bounds(positions: list[np.ndarray]) -> dict[str, list[float]]:
    """Return min/max XYZ position bounds for manifest review."""
    if not positions:
        return {"min": [], "max": []}
    array = np.asarray(positions, dtype=float)
    return {
        "min": [float(value) for value in np.min(array, axis=0)],
        "max": [float(value) for value in np.max(array, axis=0)],
    }


__all__ = [
    "DEFAULT_CAMERA_DISTANCE_M",
    "DEFAULT_CAMERA_MODE",
    "DEFAULT_CAMERA_PITCH_DEG",
    "DEFAULT_CAMERA_YAW_DEG",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_PPO_CONFIG_PATH",
    "DEFAULT_SEED",
    "SUPPORTED_CAMERA_MODES",
    "PolicyRenderResult",
    "PolicyRenderSettings",
    "default_model_path",
    "default_output_dir",
    "run_trained_policy_render",
    "run_trained_policy_render_from_paths",
]
