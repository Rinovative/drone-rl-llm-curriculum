"""Tests for tiny headless drone render-smoke helpers."""

# ruff: noqa: S101

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.experiments.cli import experiments_cli_render_smoke as cli_render_smoke
from src.experiments.rendering import experiments_rendering_smoke as render_smoke

PARSER_DURATION_SEC = 0.5
PARSER_MAX_STEPS = 4
FAKE_SIMULATOR_STEPS = 3
FALLBACK_STEPS = 5

if TYPE_CHECKING:
    import pytest


def test_render_smoke_settings_defaults_are_tiny_and_visual() -> None:
    """Verify render-smoke defaults stay bounded and visually useful."""
    settings = render_smoke.RenderSmokeSettings()

    assert settings.duration_sec == render_smoke.DEFAULT_DURATION_SEC
    assert settings.max_steps == render_smoke.DEFAULT_MAX_STEPS
    assert settings.frame_interval == render_smoke.DEFAULT_FRAME_INTERVAL
    assert settings.camera_mode == "follow_external"
    assert settings.task_shape == "circle"
    assert settings.output_dir is None


def test_cli_parser_exposes_camera_and_task_options() -> None:
    """Verify parser defaults and overrides are available without running PyBullet."""
    parser = cli_render_smoke.build_parser()
    defaults = parser.parse_args([])
    overrides = parser.parse_args(
        [
            "--duration-sec",
            "0.5",
            "--max-steps",
            "4",
            "--output-dir",
            "storage/results/render_smoke_custom",
            "--camera-mode",
            "fixed_external",
            "--task-shape",
            "line",
        ]
    )

    assert defaults.duration_sec == render_smoke.DEFAULT_DURATION_SEC
    assert defaults.max_steps == render_smoke.DEFAULT_MAX_STEPS
    assert defaults.output_dir == render_smoke.default_output_dir()
    assert defaults.output_dir.as_posix().endswith("storage/runs/render_smoke/evaluations/render_smoke")
    assert defaults.camera_mode == "follow_external"
    assert defaults.task_shape == "circle"
    assert overrides.duration_sec == PARSER_DURATION_SEC
    assert overrides.max_steps == PARSER_MAX_STEPS
    assert overrides.output_dir == Path("storage/results/render_smoke_custom")
    assert overrides.camera_mode == "fixed_external"
    assert overrides.task_shape == "line"


def test_render_smoke_writes_manifest_from_simulator_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify manifest writing with a lightweight fake simulator artifact."""

    def fake_rollout(settings: render_smoke.RenderSmokeSettings, _output_dir: Path) -> Any:
        artifact_path = _output_dir / "drone_rollout.gif"
        artifact_path.write_bytes(b"GIF89a")
        return render_smoke._RolloutArtifacts(  # noqa: SLF001
            render_mode="simulator_external_camera_gif",
            camera_mode=settings.camera_mode,
            task_shape=settings.task_shape,
            environment_backend="fake.HoverAviary",
            steps=min(settings.max_steps, FAKE_SIMULATOR_STEPS),
            output_files=(artifact_path,),
            true_simulator_rendering=True,
            warnings=(),
            positions=((0.0, 0.0, 1.0), (0.2, 0.0, 1.0)),
        )

    monkeypatch.setattr(render_smoke, "_run_simulator_rollout", fake_rollout)

    result = render_smoke.run_render_smoke(render_smoke.RenderSmokeSettings(output_dir=tmp_path, max_steps=FAKE_SIMULATOR_STEPS))
    manifest_path = tmp_path / "manifests" / render_smoke.DEFAULT_OUTPUT_FILENAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result.manifest_path == str(manifest_path)
    assert str(result.manifest["output_files"][0]).startswith(str(tmp_path / "renders"))
    assert payload["mode"] == "simulator_external_camera_gif"
    assert payload["render_mode"] == "simulator_external_camera_gif"
    assert payload["camera_mode"] == "follow_external"
    assert payload["task_shape"] == "circle"
    assert payload["environment_backend"] == "fake.HoverAviary"
    assert payload["steps"] == FAKE_SIMULATOR_STEPS
    assert payload["true_simulator_rendering"] is True
    assert payload["output_files"] == [str(tmp_path / "renders" / "drone_rollout.gif")]
    assert payload["position_bounds_xyz_m"]["max"] == [0.2, 0.0, 1.0]


def test_render_smoke_explicit_output_dir_writes_direct_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify explicit output-dir overrides preserve direct artifact placement."""
    explicit_output_dir = tmp_path / "storage" / "results" / "render_smoke_custom"

    def fail_rollout(_settings: render_smoke.RenderSmokeSettings, _output_dir: Path) -> Any:
        message = "camera unavailable"
        raise RuntimeError(message)

    monkeypatch.setattr(render_smoke, "_run_simulator_rollout", fail_rollout)

    result = render_smoke.run_render_smoke(render_smoke.RenderSmokeSettings(output_dir=explicit_output_dir, max_steps=2))

    assert result.manifest_path == str(explicit_output_dir / render_smoke.DEFAULT_OUTPUT_FILENAME)
    assert all(Path(path).parent == explicit_output_dir for path in result.manifest["output_files"])


def test_render_smoke_fallback_writes_visible_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify fallback artifact creation does not require PyBullet rendering."""

    def fail_rollout(_settings: render_smoke.RenderSmokeSettings, _output_dir: Path) -> Any:
        message = "camera unavailable"
        raise RuntimeError(message)

    monkeypatch.setattr(render_smoke, "_run_simulator_rollout", fail_rollout)

    result = render_smoke.run_render_smoke(
        render_smoke.RenderSmokeSettings(output_dir=tmp_path, max_steps=FALLBACK_STEPS, camera_mode="fixed_external", task_shape="line")
    )
    output_files = [Path(path) for path in result.manifest["output_files"]]

    assert result.manifest["mode"] == "fallback_trajectory_plot"
    assert result.manifest["camera_mode"] == "fixed_external"
    assert result.manifest["task_shape"] == "line"
    assert result.manifest["true_simulator_rendering"] is False
    assert result.manifest["steps"] == FALLBACK_STEPS
    assert result.warnings
    assert output_files
    assert all(path.exists() and path.stat().st_size > 0 for path in output_files)


def test_cli_help_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify the CLI help path is available without running the simulator."""
    try:
        cli_render_smoke.main(["--help"])
    except SystemExit as exc:
        status = exc.code
    else:
        status = 0

    captured = capsys.readouterr()
    assert status == 0
    assert "render smoke" in captured.out
    assert "--duration-sec" in captured.out
    assert "--camera-mode" in captured.out
    assert "--task-shape" in captured.out
