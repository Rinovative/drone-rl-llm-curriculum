"""Tests for continuous multi-phase scenario rendering helpers."""

# ruff: noqa: S101, SLF001, PLR2004

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.experiments.cli import experiments_cli_render_scenario as cli_render_scenario
from src.experiments.rendering import experiments_rendering_policy as policy_render
from src.experiments.rendering import experiments_rendering_scenario as scenario_render

SCRIPTED_CONFIG = Path("configs/scenarios/scripted_reference_line_polyline.yaml")
SHOWCASE_CONFIG = Path("configs/scenarios/showcase_hover_line_polyline.yaml")
CIRCLE_CONFIG = Path("configs/scenarios/scripted_reference_circle_polyline.yaml")
CONFIG_MAX_STEPS = 320
CLI_MAX_STEPS = 123
CLI_SEED = 4
CLI_CAMERA_DISTANCE = 1.4
CLI_CAMERA_YAW = 20.0
CLI_CAMERA_PITCH = -15.0
MANIFEST_TOTAL_STEPS = 2


def test_scenario_render_imports_through_package_alias() -> None:
    """Verify scenario render helpers are exposed by the experiments package."""
    assert scenario_render is not None
    assert scenario_render.ScenarioRenderSettings is not None


def test_load_scripted_scenario_settings_parses_concrete_geometry_and_holds() -> None:
    """Verify the scripted scenario config carries local primitive geometry and explicit holds."""
    settings = scenario_render.load_scenario_render_settings(SCRIPTED_CONFIG)

    assert settings.scenario_name == "scripted_reference_line_polyline"
    assert settings.task_config_path == Path("configs/smoke/trajectory_validation.yaml")
    assert settings.controller == policy_render.SCRIPTED_REFERENCE_CONTROLLER
    assert settings.max_steps == CONFIG_MAX_STEPS
    assert settings.seed == 0
    assert settings.camera_mode == "follow_external"
    assert settings.final_hold_sec == 2.0
    assert settings.phases[0].name == "line_forward"
    assert settings.phases[0].phase_type == "line"
    assert settings.phases[0].task_shape == "line"
    assert settings.phases[0].duration_sec == 10.0
    assert settings.phases[0].hold_after_sec == 1.0
    assert settings.phases[0].geometry["delta_position"] == [0.6, 0.0, 0.0]
    assert settings.phases[0].start_mode == scenario_render.START_MODE_INITIAL
    assert settings.phases[1].phase_type == "polyline"
    assert settings.phases[1].duration_sec == 16.0
    assert settings.phases[1].geometry["waypoints"][-1] == [0.4, 0.4, 0.0]
    assert settings.phases[1].start_mode == scenario_render.START_MODE_PREVIOUS_END


def test_load_showcase_scenario_settings_requires_ppo_model_run() -> None:
    """Verify the PPO showcase config carries its model-run source."""
    settings = scenario_render.load_scenario_render_settings(SHOWCASE_CONFIG)

    assert settings.scenario_name == "showcase_hover_line_polyline"
    assert settings.controller == policy_render.PPO_CONTROLLER
    assert settings.model_run_name == "ppo_line_100k_seed0"
    assert settings.phases[0].phase_type == "hover"
    assert settings.phases[0].geometry["hold_current_position"] is True


def test_ppo_scenario_settings_reject_missing_model_run_name() -> None:
    """Verify PPO scenario rendering cannot silently fall back to old model paths."""
    with pytest.raises(ValueError, match="model_run_name"):
        scenario_render.ScenarioRenderSettings(
            scenario_name="bad_ppo",
            controller=policy_render.PPO_CONTROLLER,
            phases=(scenario_render.ScenarioPhase(name="line", phase_type="line"),),
        )


def test_compose_line_phase_uses_custom_delta_position() -> None:
    """Verify a line phase uses scenario delta_position instead of catalog endpoints."""
    phase = scenario_render.ScenarioPhase(
        name="short_line",
        phase_type="line",
        duration_sec=2.0,
        geometry={"delta_position": [0.25, 0.75, 0.0]},
    )
    settings = scenario_render.ScenarioRenderSettings(
        scenario_name="line_delta_test",
        phases=(phase,),
        max_steps=30,
    )

    composition = scenario_render.compose_scenario_reference(settings)

    assert np.allclose(composition.phase_start_positions[0], [0.0, 0.0, 1.0])
    assert np.allclose(composition.phase_end_positions[0], [0.25, 0.75, 1.0])
    assert composition.phase_geometry[0]["delta_position"] == [0.25, 0.75, 0.0]


def test_compose_polyline_phase_uses_custom_local_waypoints_and_holds() -> None:
    """Verify a polyline phase follows scenario local waypoints and explicit holds."""
    settings = scenario_render.load_scenario_render_settings(SCRIPTED_CONFIG)
    composition = scenario_render.compose_scenario_reference(settings)

    assert composition.reference.shape == "scenario"
    assert composition.reference.positions.shape == (292, 3)
    assert np.all(np.diff(composition.reference.times) > 0.0)
    assert composition.phase_step_ranges == ({"start": 0, "end": 101}, {"start": 111, "end": 272})
    assert composition.phase_hold_step_ranges == ({"phase_index": 0, "phase_name": "line_forward", "phase_type": "line", "start": 101, "end": 111},)
    assert composition.reference_motion_steps == 262
    assert composition.reference_motion_end_step == 272
    assert composition.phase_hold_steps == 10
    assert composition.final_hold_steps == 20
    assert composition.total_reference_steps == 292
    assert composition.final_hold_step_range == {"start": 272, "end": 292}
    assert composition.phase_offsets[0] == [0.0, 0.0, 0.0]
    assert composition.phase_offsets[1] == [0.6, 0.0, 0.0]
    assert composition.phase_end_positions[0] == composition.phase_start_positions[1]
    assert composition.reference.positions[100].tolist() == composition.reference.positions[101].tolist()
    assert composition.reference.positions[110].tolist() == composition.reference.positions[111].tolist()
    assert np.isclose(np.max(composition.reference.positions[:, 2]), 1.0)
    assert composition.phase_geometry[1]["waypoints"] == [[0.0, 0.0, 0.0], [0.0, 0.4, 0.0], [0.4, 0.4, 0.0]]


def test_compose_circle_phase_uses_custom_radius_and_polyline_differs() -> None:
    """Verify configurable circle geometry and a different custom polyline scenario."""
    settings = scenario_render.load_scenario_render_settings(CIRCLE_CONFIG)
    composition = scenario_render.compose_scenario_reference(settings)

    assert composition.reference.positions.shape == (182, 3)
    assert composition.phase_step_ranges == ({"start": 0, "end": 81}, {"start": 81, "end": 162})
    assert composition.reference_motion_steps == 162
    assert composition.phase_hold_steps == 0
    assert composition.final_hold_steps == 20
    assert composition.total_reference_steps == 182
    assert composition.phase_geometry[0]["type"] == "circle"
    assert composition.phase_geometry[0]["radius_m"] == 0.25
    assert composition.phase_geometry[0]["center_offset"] == [0.0, 0.25, 0.0]
    assert composition.phase_geometry[0]["direction"] == "ccw"
    assert np.allclose(composition.phase_end_positions[0], composition.phase_start_positions[1])
    assert composition.phase_geometry[1]["waypoints"] == [[0.0, 0.0, 0.0], [0.35, -0.15, 0.0], [0.65, 0.25, 0.0]]
    assert not np.allclose(composition.phase_end_positions[1], [1.5, 0.5, 1.0])


def test_effective_max_steps_cover_reference_or_derive_with_margin() -> None:
    """Verify max_steps is a safety cap and omitted caps derive from reference length."""
    configured = scenario_render.load_scenario_render_settings(SCRIPTED_CONFIG)
    composition = scenario_render.compose_scenario_reference(configured)

    assert scenario_render._effective_max_steps(configured, composition) == CONFIG_MAX_STEPS

    derived = scenario_render.ScenarioRenderSettings(
        scenario_name="derive_cap",
        phases=(
            scenario_render.ScenarioPhase(
                name="line",
                phase_type="line",
                duration_sec=2.0,
                geometry={"delta_position": [0.2, 0.0, 0.0]},
            ),
        ),
        max_steps=None,
    )
    derived_composition = scenario_render.compose_scenario_reference(derived)
    assert scenario_render._effective_max_steps(derived, derived_composition) == (
        derived_composition.total_reference_steps + scenario_render.DEFAULT_MAX_STEP_SAFETY_MARGIN
    )


def test_max_steps_smaller_than_reference_fails_clearly() -> None:
    """Verify too-small safety caps fail before producing incomplete scenario rollouts."""
    settings = scenario_render.load_scenario_render_settings(SCRIPTED_CONFIG)
    too_small = scenario_render.ScenarioRenderSettings(
        scenario_config_path=settings.scenario_config_path,
        scenario_name=settings.scenario_name,
        task_config_path=settings.task_config_path,
        phases=settings.phases,
        controller=settings.controller,
        max_steps=100,
        seed=settings.seed,
        final_hold_sec=settings.final_hold_sec,
    )
    composition = scenario_render.compose_scenario_reference(too_small)

    with pytest.raises(ValueError, match="smaller than the composed scenario reference length"):
        scenario_render._effective_max_steps(too_small, composition)


def test_base_time_limit_covers_composed_scenario_duration() -> None:
    """Verify scenario rendering configures the base env timeout from composed duration."""
    settings = scenario_render.load_scenario_render_settings(SCRIPTED_CONFIG)
    composition = scenario_render.compose_scenario_reference(settings)

    assert scenario_render._base_time_limit_sec(composition) > composition.scenario_duration_sec
    assert composition.scenario_duration_sec >= 29.0


def test_phase_composition_rejects_discontinuous_initial_boundary() -> None:
    """Verify adjacent phase boundaries must be continuous."""
    settings = scenario_render.ScenarioRenderSettings(
        scenario_name="bad_discontinuous",
        phases=(
            scenario_render.ScenarioPhase(
                name="line_a",
                phase_type="line",
                duration_sec=2.0,
                start_mode="initial",
                geometry={"delta_position": [0.5, 0.0, 0.0]},
            ),
            scenario_render.ScenarioPhase(
                name="line_b",
                phase_type="line",
                duration_sec=2.0,
                start_mode="initial",
                geometry={"delta_position": [0.25, 0.0, 0.0]},
            ),
        ),
    )

    with pytest.raises(ValueError, match="discontinuous"):
        scenario_render.compose_scenario_reference(settings)


def test_add_phase_fields_labels_trace_rows_with_required_aliases() -> None:
    """Verify scenario trace enrichment adds phase and hold metadata aliases."""
    settings = scenario_render.load_scenario_render_settings(SCRIPTED_CONFIG)
    composition = scenario_render.compose_scenario_reference(settings)
    records = [
        {
            "step_index": 0,
            "reference_position_xyz_m": [0.0, 0.0, 1.0],
            "actual_position_xyz_m": [0.0, 0.0, 1.0],
            "position_error_m": 0.0,
        },
        {
            "step_index": 101,
            "reference_position_xyz_m": [1.0, 0.0, 1.0],
            "actual_position_xyz_m": [1.0, 0.0, 1.0],
            "position_error_m": 0.0,
        },
        {
            "step_index": 111,
            "reference_position_xyz_m": [1.0, 0.0, 1.0],
            "actual_position_xyz_m": [1.0, 0.0, 1.0],
            "position_error_m": 0.0,
        },
        {
            "step_index": 272,
            "reference_position_xyz_m": [1.0, 0.4, 1.0],
            "actual_position_xyz_m": [1.4, 0.35, 1.0],
            "position_error_m": 0.1118,
        },
    ]

    enriched = scenario_render._add_phase_fields(records, composition)

    assert enriched[0]["global_step"] == 0
    assert enriched[0]["phase_index"] == 0
    assert enriched[0]["phase_name"] == "line_forward"
    assert enriched[0]["phase_type"] == "line"
    assert enriched[0]["phase_task_shape"] == "line"
    assert enriched[0]["is_phase_hold"] is False
    assert enriched[0]["is_final_hold"] is False
    assert enriched[0]["reference_position"] == [0.0, 0.0, 1.0]
    assert enriched[0]["current_position"] == [0.0, 0.0, 1.0]
    assert enriched[0]["position_error"] == 0.0
    assert enriched[1]["phase_index"] == 0
    assert enriched[1]["is_phase_hold"] is True
    assert enriched[1]["is_final_hold"] is False
    assert enriched[2]["phase_index"] == 1
    assert enriched[2]["phase_type"] == "polyline"
    assert enriched[2]["is_phase_hold"] is False
    assert enriched[3]["phase_name"] == "final_hold"
    assert enriched[3]["phase_type"] == "final_hold"
    assert enriched[3]["is_final_hold"] is True


def test_scenario_run_name_derivation_matches_contract() -> None:
    """Verify default evaluation run names follow the scenario contract."""
    scripted = scenario_render.load_scenario_render_settings(SCRIPTED_CONFIG)
    ppo = scenario_render.load_scenario_render_settings(SHOWCASE_CONFIG)

    assert scenario_render._evaluation_run_name(scripted) == "eval_scripted_reference_on_line_polyline"
    assert scenario_render._evaluation_run_name(ppo) == "eval_ppo_line_100k_seed0_on_showcase_hover_line_polyline"
    circle = scenario_render.load_scenario_render_settings(CIRCLE_CONFIG)
    assert scenario_render._evaluation_run_name(circle) == "eval_scripted_reference_on_circle_polyline"


def test_cli_parser_accepts_scenario_overrides() -> None:
    """Verify the scenario CLI exposes requested override arguments."""
    parser = cli_render_scenario.build_parser()
    args = parser.parse_args(
        [
            "--config",
            str(SCRIPTED_CONFIG),
            "--run-name",
            "custom_scenario_eval",
            "--controller",
            "scripted_reference",
            "--model-run-name",
            "ppo_line_100k_seed0",
            "--max-steps",
            str(CLI_MAX_STEPS),
            "--seed",
            str(CLI_SEED),
            "--camera-mode",
            "fixed_external",
            "--camera-distance",
            str(CLI_CAMERA_DISTANCE),
            "--camera-yaw",
            str(CLI_CAMERA_YAW),
            "--camera-pitch",
            str(CLI_CAMERA_PITCH),
        ]
    )

    assert args.config == SCRIPTED_CONFIG
    assert args.run_name == "custom_scenario_eval"
    assert args.controller == "scripted_reference"
    assert args.model_run_name == "ppo_line_100k_seed0"
    assert args.max_steps == CLI_MAX_STEPS
    assert args.seed == CLI_SEED
    assert args.camera_mode == "fixed_external"
    assert args.camera_distance == CLI_CAMERA_DISTANCE
    assert args.camera_yaw == CLI_CAMERA_YAW
    assert args.camera_pitch == CLI_CAMERA_PITCH


def test_build_scenario_manifest_includes_required_fields(tmp_path: Path) -> None:
    """Verify scenario manifests include phase geometry, holds, reset count, and completion metadata."""
    settings = scenario_render.load_scenario_render_settings(SCRIPTED_CONFIG)
    composition = scenario_render.compose_scenario_reference(settings)
    effective_max_steps = scenario_render._effective_max_steps(settings, composition)
    base_time_limit_sec = scenario_render._base_time_limit_sec(composition)
    metrics = {"run_type": "evaluation", "evaluation_type": "scenario", "total_steps": MANIFEST_TOTAL_STEPS}
    completion = scenario_render._scenario_completion_summary(
        actual_steps=composition.total_reference_steps,
        requested_max_steps=settings.max_steps,
        effective_max_steps=effective_max_steps,
        reference_sample_count=composition.total_reference_steps,
        reference_motion_steps=composition.reference_motion_steps,
        reference_motion_end_step=composition.reference_motion_end_step,
        phase_hold_steps=composition.phase_hold_steps,
        phase_hold_end_step=composition.phase_hold_end_step,
        final_hold_steps=composition.final_hold_steps,
        terminated=True,
        truncated=False,
        termination_reason="tracking_reference_complete",
    )

    payload = scenario_render._build_scenario_manifest(
        settings=settings,
        evaluation_run_name="eval_scripted_reference_on_line_polyline",
        scenario_name="scripted_reference_line_polyline",
        composition=composition,
        model_path=None,
        training_task_shape=None,
        gif_path=tmp_path / "renders" / "scenario_rollout.gif",
        trace_path=tmp_path / "traces" / "scenario_rollout_trace.jsonl",
        plot_paths={"trajectory_xy": str(tmp_path / "plots" / "trajectory_xy.png")},
        actual_steps=composition.total_reference_steps,
        requested_max_steps=settings.max_steps,
        effective_max_steps=effective_max_steps,
        base_time_limit_sec=base_time_limit_sec,
        completion=completion,
        termination_reason="tracking_reference_complete",
        true_simulator_rendering=True,
        policy_predict_used=False,
        metrics=metrics,
        warnings=(),
        final_info={
            "current_position": [1.0, 0.0, 1.0],
            "reference_position": [1.0, 0.0, 1.0],
            "position_error_m": 0.0,
        },
        final_action=[[1.0, 0.0, 1.0]],
        actual_positions=[
            np.array([0.0, 0.0, 1.0]),
            np.array([0.8, -0.2, 1.1]),
            np.array([1.2, 0.4, 0.9]),
        ],
        rollout_reference_positions=[np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.4, 1.0])],
        output_dir=tmp_path,
    )

    assert payload["run_type"] == "evaluation"
    assert payload["evaluation_type"] == "scenario"
    assert payload["scenario_name"] == "scripted_reference_line_polyline"
    assert payload["controller_type"] == "scripted_reference"
    assert payload["model_run_name"] is None
    assert payload["policy_predict_used"] is False
    assert payload["true_simulator_rendering"] is True
    assert payload["task_config_path"] == "configs/smoke/trajectory_validation.yaml"
    assert payload["scenario_config_path"] == str(SCRIPTED_CONFIG)
    assert payload["phase_names"] == ["line_forward", "polyline_turns"]
    assert payload["phase_types"] == ["line", "polyline"]
    assert payload["phase_task_shapes"] == ["line", "polyline"]
    assert payload["phase_offsets"][1] == [0.6, 0.0, 0.0]
    assert payload["phase_geometry"][0]["delta_position"] == [0.6, 0.0, 0.0]
    assert payload["phase_geometry"][1]["waypoints"][-1] == [0.4, 0.4, 0.0]
    assert payload["phase_hold_step_ranges"] == [{"phase_index": 0, "phase_name": "line_forward", "phase_type": "line", "start": 101, "end": 111}]
    assert payload["scenario_duration_sec"] >= 29.0
    assert payload["total_steps"] == composition.total_reference_steps
    assert payload["requested_max_steps"] == CONFIG_MAX_STEPS
    assert payload["effective_max_steps"] == CONFIG_MAX_STEPS
    assert payload["base_time_limit_sec"] > payload["scenario_duration_sec"]
    assert payload["reference_motion_steps"] == 262
    assert payload["phase_hold_steps"] == 10
    assert payload["final_hold_steps"] == 20
    assert payload["total_reference_steps"] == 292
    assert payload["final_hold_sec"] == 2.0
    assert payload["final_hold_step_range"] == {"start": 272, "end": 292}
    assert payload["completed_reference_motion"] is True
    assert payload["completed_phase_holds"] is True
    assert payload["completed_final_hold"] is True
    assert payload["completed_reference"] is True
    assert payload["ended_normally"] is True
    assert payload["survived_fraction"] == 1.0
    assert 0.0 < payload["rollout_step_fraction"] < 1.0
    assert payload["reference_completion_fraction"] == 1.0
    assert payload["position_bounds"] == {"min": [0.0, -0.2, 0.9], "max": [1.2, 0.4, 1.1]}
    assert payload["reference_position_bounds"]["max"] == [1.0, 0.4, 1.0]
    assert payload["termination_reason"] == "tracking_reference_complete"
    assert payload["reset_count"] == 1
    assert payload["gif_path"].endswith("renders/scenario_rollout.gif")
    assert payload["trace_path"].endswith("traces/scenario_rollout_trace.jsonl")
    assert payload["plot_paths"]["trajectory_xy"].endswith("trajectory_xy.png")
