"""
===============================================================================
experiments_rendering_smoke.py
===============================================================================
Run tiny headless render smoke rollouts with gym-pybullet-drones when available.

Responsibilities:
  - Start the upstream hover simulator for a bounded no-training rollout
  - Save visible third-person render artifacts and a JSON manifest under results
  - Provide a deterministic visual fallback when simulator camera capture is unavailable

Design principles:
  - Keep defaults tiny, headless, deterministic, and reviewable
  - Treat rendering as an integration smoke test, not as an RL training loop
  - Keep optional rendering dependencies lazy and isolated

Boundaries:
  - PPO training and trajectory-tracking Gym wrappers belong in later experiment modules
  - Docker runner behavior belongs in scripts and is not modified here
===============================================================================

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src import envs, utils

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_DURATION_SEC = 2.0
DEFAULT_MAX_STEPS = 60
DEFAULT_OUTPUT_FILENAME = "render_manifest.json"
DEFAULT_FRAME_INTERVAL = 5
DEFAULT_IMAGE_WIDTH = 480
DEFAULT_IMAGE_HEIGHT = 360
DEFAULT_SEED = 7
DEFAULT_CAMERA_MODE = "follow_external"
DEFAULT_TASK_SHAPE = "circle"
SUPPORTED_CAMERA_MODES = ("follow_external", "fixed_external")
SUPPORTED_TASK_SHAPES = ("hover", "line", "circle")

_CAMERA_DISTANCE_M = 1.8
_CAMERA_YAW_DEG = 45.0
_CAMERA_PITCH_DEG = -32.0
_CAMERA_FOV_DEG = 52.0
_TRAJECTORY_RADIUS_M = 0.45
_TRAJECTORY_ALTITUDE_M = 1.0


@dataclass(frozen=True)
class RenderSmokeSettings:
    """
    Settings for a tiny headless drone render smoke run.

    Parameters
    ----------
    duration_sec
        Maximum simulated duration to run, in seconds.
    max_steps
        Hard cap on environment steps.
    output_dir
        Directory where the manifest and visible artifacts are written.
    seed
        Deterministic environment reset seed.
    frame_interval
        Number of simulator steps between captured frames.
    image_width
        Captured frame width in pixels.
    image_height
        Captured frame height in pixels.
    camera_mode
        External camera mode. Supported values are ``follow_external`` and ``fixed_external``.
    task_shape
        Small deterministic render trajectory. Supported values are ``hover``, ``line``, and ``circle``.

    """

    duration_sec: float = DEFAULT_DURATION_SEC
    max_steps: int = DEFAULT_MAX_STEPS
    output_dir: Path | None = None
    seed: int = DEFAULT_SEED
    frame_interval: int = DEFAULT_FRAME_INTERVAL
    image_width: int = DEFAULT_IMAGE_WIDTH
    image_height: int = DEFAULT_IMAGE_HEIGHT
    camera_mode: str = DEFAULT_CAMERA_MODE
    task_shape: str = DEFAULT_TASK_SHAPE

    def __post_init__(self) -> None:
        """Validate render smoke settings."""
        if not np.isfinite(self.duration_sec) or self.duration_sec <= 0.0:
            message = "duration_sec must be finite and positive"
            raise ValueError(message)
        if self.max_steps <= 0:
            message = "max_steps must be positive"
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
        if self.task_shape not in SUPPORTED_TASK_SHAPES:
            message = f"task_shape must be one of: {', '.join(SUPPORTED_TASK_SHAPES)}"
            raise ValueError(message)


@dataclass(frozen=True)
class RenderSmokeResult:
    """
    Summary returned by a render smoke run.

    Parameters
    ----------
    manifest_path
        Path to the written JSON manifest.
    manifest
        JSON-serializable manifest describing the run and generated artifacts.
    warnings
        Nonfatal warnings, including simulator or encoding fallbacks.

    """

    manifest_path: str
    manifest: dict[str, Any]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _RolloutArtifacts:
    """Internal representation of visible rollout artifacts before manifest writing."""

    render_mode: str
    camera_mode: str
    task_shape: str
    environment_backend: str
    steps: int
    output_files: tuple[Path, ...]
    true_simulator_rendering: bool
    warnings: tuple[str, ...]
    positions: tuple[tuple[float, float, float], ...]


def default_output_dir() -> Path:
    """Return the default render smoke directory under the canonical run layout."""
    return utils.artifacts.get_run_evaluation_dir("render_smoke", "render_smoke")


def run_render_smoke(settings: RenderSmokeSettings | None = None) -> RenderSmokeResult:
    """
    Run a tiny drone render smoke rollout and write visible artifacts.

    Parameters
    ----------
    settings
        Optional smoke-run settings. Defaults are used when omitted.

    Returns
    -------
    RenderSmokeResult
        Manifest path, manifest payload, and nonfatal warnings.

    """
    active_settings = settings or RenderSmokeSettings()
    output_dir = active_settings.output_dir or default_output_dir()
    renders_dir, manifests_dir = _artifact_dirs(output_dir)
    renders_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    try:
        artifacts = _run_simulator_rollout(active_settings, renders_dir)
    except Exception as exc:  # noqa: BLE001 - the smoke path must degrade to a visible fallback artifact.
        warning = f"simulator external camera render unavailable; wrote fallback trajectory plot instead: {exc}"
        artifacts = _write_fallback_plot(active_settings, renders_dir, warnings=(warning,))

    manifest = _build_manifest(artifacts)
    manifest_path = manifests_dir / DEFAULT_OUTPUT_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return RenderSmokeResult(manifest_path=str(manifest_path), manifest=manifest, warnings=artifacts.warnings)


def _artifact_dirs(output_dir: Path) -> tuple[Path, Path]:
    """Return render and manifest directories for a run-root or explicit output override."""
    if "results" in output_dir.parts:
        return output_dir, output_dir
    return output_dir / "renders", output_dir / "manifests"


def _run_simulator_rollout(settings: RenderSmokeSettings, output_dir: Path) -> _RolloutArtifacts:
    """Run the upstream hover simulator and capture external third-person frames."""
    environment_backend = "gym_pybullet_drones.envs.HoverAviary"
    simulator_env: Any = envs.builders.make_hover_aviary_env(gui=False, record=False)
    positions: list[tuple[float, float, float]] = []
    frames: list[np.ndarray] = []
    warnings: list[str] = []
    try:
        simulator_env.reset(seed=settings.seed)
        ctrl_freq = int(getattr(simulator_env, "CTRL_FREQ", 30))
        step_limit = min(settings.max_steps, max(1, int(np.ceil(settings.duration_sec * ctrl_freq))))
        zero_action = np.zeros(simulator_env.action_space.shape, dtype=simulator_env.action_space.dtype)

        for step_index in range(step_limit + 1):
            progress = step_index / max(step_limit, 1)
            position, yaw_rad = _trajectory_pose(settings.task_shape, progress)
            _place_drone_for_render(simulator_env, position, yaw_rad)
            _append_position(positions, position)
            if step_index % settings.frame_interval == 0 or step_index == step_limit:
                _capture_external_frame(frames, simulator_env, settings, position)
            if step_index < step_limit:
                simulator_env.step(zero_action)
    finally:
        simulator_env.close()

    if not frames:
        message = "simulator produced no external camera frames"
        raise RuntimeError(message)

    output_files, artifact_mode, encode_warning = _write_frame_artifacts(frames, output_dir, settings, environment_backend)
    if encode_warning:
        warnings.append(encode_warning)

    return _RolloutArtifacts(
        render_mode=artifact_mode,
        camera_mode=settings.camera_mode,
        task_shape=settings.task_shape,
        environment_backend=environment_backend,
        steps=max(len(positions) - 1, 0),
        output_files=tuple(output_files),
        true_simulator_rendering=True,
        warnings=tuple(warnings),
        positions=tuple(positions),
    )


def _trajectory_pose(task_shape: str, progress: float) -> tuple[np.ndarray, float]:
    """Return a deterministic XYZ position and yaw for a small visual render path."""
    bounded_progress = float(np.clip(progress, 0.0, 1.0))
    if task_shape == "hover":
        return np.array([0.0, 0.0, _TRAJECTORY_ALTITUDE_M], dtype=float), 0.0
    if task_shape == "line":
        x_position = -_TRAJECTORY_RADIUS_M + (2.0 * _TRAJECTORY_RADIUS_M * bounded_progress)
        return np.array([x_position, 0.0, _TRAJECTORY_ALTITUDE_M], dtype=float), 0.0

    angle = 2.0 * np.pi * bounded_progress
    position = np.array(
        [
            _TRAJECTORY_RADIUS_M * np.cos(angle),
            _TRAJECTORY_RADIUS_M * np.sin(angle),
            _TRAJECTORY_ALTITUDE_M,
        ],
        dtype=float,
    )
    return position, float(angle + (np.pi / 2.0))


def _place_drone_for_render(simulator_env: Any, position: np.ndarray, yaw_rad: float) -> None:
    """Place the simulated drone on the deterministic smoke path for visual capture."""
    import pybullet as p  # noqa: PLC0415

    orientation = p.getQuaternionFromEuler([0.0, 0.0, yaw_rad])
    drone_id = int(simulator_env.DRONE_IDS[0])
    p.resetBasePositionAndOrientation(drone_id, position.tolist(), orientation, physicsClientId=simulator_env.CLIENT)
    p.resetBaseVelocity(drone_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=simulator_env.CLIENT)
    _sync_simulator_pose_cache(simulator_env, position, orientation, yaw_rad)


def _sync_simulator_pose_cache(simulator_env: Any, position: np.ndarray, orientation: tuple[float, float, float, float], yaw_rad: float) -> None:
    """Update upstream pose arrays used by render helpers after deterministic placement."""
    simulator_env.pos[0, :3] = position
    simulator_env.quat[0, :4] = np.asarray(orientation, dtype=float)
    simulator_env.rpy[0, :3] = np.array([0.0, 0.0, yaw_rad], dtype=float)
    simulator_env.vel[0, :3] = np.zeros(3, dtype=float)
    simulator_env.ang_v[0, :3] = np.zeros(3, dtype=float)


def _append_position(positions: list[tuple[float, float, float]], position: np.ndarray) -> None:
    """Append an XYZ position to the manifest trajectory summary."""
    positions.append((float(position[0]), float(position[1]), float(position[2])))


def _capture_external_frame(frames: list[np.ndarray], simulator_env: Any, settings: RenderSmokeSettings, position: np.ndarray) -> None:
    """Capture one RGB frame from a headless external PyBullet camera."""
    import pybullet as p  # noqa: PLC0415

    target = _camera_target(settings.camera_mode, position)
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target.tolist(),
        distance=_CAMERA_DISTANCE_M,
        yaw=_CAMERA_YAW_DEG,
        pitch=_CAMERA_PITCH_DEG,
        roll=0.0,
        upAxisIndex=2,
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=_CAMERA_FOV_DEG,
        aspect=settings.image_width / settings.image_height,
        nearVal=0.05,
        farVal=20.0,
    )
    _, _, rgb, _, _ = p.getCameraImage(
        width=settings.image_width,
        height=settings.image_height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        flags=p.ER_NO_SEGMENTATION_MASK,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=simulator_env.CLIENT,
    )
    frames.append(np.reshape(np.asarray(rgb, dtype=np.uint8), (settings.image_height, settings.image_width, 4))[..., :3])


def _camera_target(camera_mode: str, position: np.ndarray) -> np.ndarray:
    """Return the world-space target for a supported external camera mode."""
    flight_center = np.array([0.0, 0.0, _TRAJECTORY_ALTITUDE_M], dtype=float)
    if camera_mode == "fixed_external":
        return flight_center
    return (0.55 * flight_center) + (0.45 * position)


def _write_frame_artifacts(
    frames: list[np.ndarray],
    output_dir: Path,
    settings: RenderSmokeSettings,
    environment_backend: str,
) -> tuple[tuple[Path, ...], str, str | None]:
    """Write simulator frames as a GIF when possible, otherwise as PNG frames."""
    try:
        import imageio.v2 as imageio  # noqa: PLC0415

        output_path = output_dir / "drone_rollout.gif"
        frame_duration = max(settings.frame_interval / 30.0, 0.05)
        frame_payload: Any = frames
        imageio.mimsave(output_path, frame_payload, duration=frame_duration)
    except Exception as exc:  # noqa: BLE001 - PNG frames are the intended media fallback.
        frame_paths = _write_png_frames(frames, output_dir)
        warning = f"GIF encoding unavailable for {environment_backend}; saved PNG frame sequence instead: {exc}"
        return frame_paths, "simulator_external_camera_png_frames", warning
    else:
        return (output_path,), "simulator_external_camera_gif", None


def _write_png_frames(frames: list[np.ndarray], output_dir: Path) -> tuple[Path, ...]:
    """Write RGB frame arrays as a PNG sequence."""
    from PIL import Image  # noqa: PLC0415

    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for frame_index, frame in enumerate(frames):
        path = frame_dir / f"frame_{frame_index:04d}.png"
        Image.fromarray(frame).save(path)
        paths.append(path)
    return tuple(paths)


def _write_fallback_plot(settings: RenderSmokeSettings, output_dir: Path, warnings: tuple[str, ...]) -> _RolloutArtifacts:
    """Write a visible deterministic trajectory plot when simulator rendering is unavailable."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    step_count = settings.max_steps
    positions = tuple(_trajectory_pose(settings.task_shape, step_index / max(step_count, 1))[0] for step_index in range(step_count + 1))
    x_positions = np.array([position[0] for position in positions], dtype=float)
    y_positions = np.array([position[1] for position in positions], dtype=float)

    output_path = output_dir / "drone_rollout_fallback.png"
    figure, axes = plt.subplots(figsize=(6, 4), dpi=120)
    axes.plot(x_positions, y_positions, marker="o", markersize=2.5, linewidth=1.5)
    axes.set_title(f"Render smoke fallback: {settings.task_shape}")
    axes.set_xlabel("x position (m)")
    axes.set_ylabel("y position (m)")
    axes.set_aspect("equal", adjustable="box")
    axes.grid(True, alpha=0.35)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)

    return _RolloutArtifacts(
        render_mode="fallback_trajectory_plot",
        camera_mode=settings.camera_mode,
        task_shape=settings.task_shape,
        environment_backend="matplotlib_deterministic_fallback",
        steps=step_count,
        output_files=(output_path,),
        true_simulator_rendering=False,
        warnings=warnings,
        positions=tuple((float(position[0]), float(position[1]), float(position[2])) for position in positions),
    )


def _build_manifest(artifacts: _RolloutArtifacts) -> dict[str, Any]:
    """Build the render smoke manifest payload."""
    bounds = _position_bounds(artifacts.positions)
    return {
        "mode": artifacts.render_mode,
        "render_mode": artifacts.render_mode,
        "camera_mode": artifacts.camera_mode,
        "task_shape": artifacts.task_shape,
        "environment_backend": artifacts.environment_backend,
        "steps": artifacts.steps,
        "output_files": [str(path) for path in artifacts.output_files],
        "true_simulator_rendering": artifacts.true_simulator_rendering,
        "warnings": list(artifacts.warnings),
        "fallbacks": list(artifacts.warnings) if not artifacts.true_simulator_rendering else [],
        "final_position_xyz_m": list(artifacts.positions[-1]) if artifacts.positions else [],
        "position_bounds_xyz_m": bounds,
    }


def _position_bounds(positions: tuple[tuple[float, float, float], ...]) -> dict[str, list[float]]:
    """Return min/max XYZ bounds for manifest review."""
    if not positions:
        return {"min": [], "max": []}
    position_array = np.asarray(positions, dtype=float)
    return {
        "min": [float(value) for value in np.min(position_array, axis=0)],
        "max": [float(value) for value in np.max(position_array, axis=0)],
    }


__all__ = [
    "DEFAULT_CAMERA_MODE",
    "DEFAULT_DURATION_SEC",
    "DEFAULT_FRAME_INTERVAL",
    "DEFAULT_IMAGE_HEIGHT",
    "DEFAULT_IMAGE_WIDTH",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_OUTPUT_FILENAME",
    "DEFAULT_SEED",
    "DEFAULT_TASK_SHAPE",
    "SUPPORTED_CAMERA_MODES",
    "SUPPORTED_TASK_SHAPES",
    "RenderSmokeResult",
    "RenderSmokeSettings",
    "default_output_dir",
    "run_render_smoke",
]
