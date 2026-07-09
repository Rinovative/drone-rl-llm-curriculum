"""Tests for deterministic LLM task schema helpers."""

# ruff: noqa: S101

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from src import llm, validation


def _valid_tasks_by_shape() -> list[dict[str, object]]:
    """Return valid tasks covering all supported shape-specific keys."""
    contracts = validation.contracts
    return [
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_HOVER_STABILIZATION,
            contracts.FIELD_DURATION_SEC: 2.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_POSITION: [0.0, 0.0, 1.0],
        },
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_NEARBY_TARGET_HOVER,
            contracts.FIELD_DURATION_SEC: 2.5,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_POSITION: [0.15, 0.0, 1.0],
        },
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_START_HOLD_THEN_SHORT_LINE,
            contracts.FIELD_HOLD_DURATION_SEC: 1.0,
            contracts.FIELD_MOVE_DURATION_SEC: 3.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_START_HOLD_ENABLED: True,
            contracts.FIELD_START_HOLD_SEC: 1.0,
            contracts.FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS: True,
            contracts.FIELD_START: [0.0, 0.0, 1.0],
            contracts.FIELD_END: [0.25, 0.0, 1.0],
        },
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_SHORT_SLOW_LINE,
            contracts.FIELD_DURATION_SEC: 4.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_START_HOLD_ENABLED: True,
            contracts.FIELD_START_HOLD_SEC: 1.0,
            contracts.FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS: True,
            contracts.FIELD_START: [0.0, 0.0, 1.0],
            contracts.FIELD_END: [0.5, 0.0, 1.0],
        },
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_HOVER,
            contracts.FIELD_DURATION_SEC: 2.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 5.0,
            contracts.FIELD_POSITION: [0.0, 0.0, 1.0],
        },
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_CIRCLE,
            contracts.FIELD_DURATION_SEC: 8.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_RADIUS: 0.5,
            contracts.FIELD_HEIGHT: 1.0,
            contracts.FIELD_CENTER: [0.0, 0.0],
            contracts.FIELD_CLOCKWISE: True,
            contracts.FIELD_START_HOLD_ENABLED: True,
            contracts.FIELD_START_HOLD_SEC: 1.0,
            contracts.FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS: True,
        },
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_ELLIPSE,
            contracts.FIELD_DURATION_SEC: 12.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_RADIUS_X: 0.35,
            contracts.FIELD_RADIUS_Y: 0.20,
            contracts.FIELD_HEIGHT: 1.0,
            contracts.FIELD_CENTER: [0.0, 0.0],
            contracts.FIELD_CLOCKWISE: False,
            contracts.FIELD_START_HOLD_ENABLED: True,
            contracts.FIELD_START_HOLD_SEC: 1.0,
            contracts.FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS: True,
        },
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_FIGURE_EIGHT,
            contracts.FIELD_DURATION_SEC: 14.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_RADIUS_X: 0.30,
            contracts.FIELD_RADIUS_Y: 0.18,
            contracts.FIELD_HEIGHT: 1.0,
            contracts.FIELD_CENTER: [0.0, 0.0],
            contracts.FIELD_CLOCKWISE: False,
            contracts.FIELD_START_HOLD_ENABLED: True,
            contracts.FIELD_START_HOLD_SEC: 1.0,
            contracts.FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS: True,
        },
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_LINE,
            contracts.FIELD_DURATION_SEC: 3.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_START_HOLD_ENABLED: True,
            contracts.FIELD_START_HOLD_SEC: 1.0,
            contracts.FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS: True,
            contracts.FIELD_START: [0.0, 0.0, 1.0],
            contracts.FIELD_END: [0.5, 0.0, 1.0],
        },
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_VERTICAL,
            contracts.FIELD_DURATION_SEC: 3.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_START_HOLD_ENABLED: True,
            contracts.FIELD_START_HOLD_SEC: 1.0,
            contracts.FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS: True,
            contracts.FIELD_XY: [0.25, -0.25],
            contracts.FIELD_START_HEIGHT: 0.8,
            contracts.FIELD_END_HEIGHT: 1.4,
        },
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_POLYLINE,
            contracts.FIELD_DURATION_SEC: 6.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_START_HOLD_ENABLED: True,
            contracts.FIELD_START_HOLD_SEC: 1.0,
            contracts.FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS: True,
            contracts.FIELD_POINTS: [[0.0, 0.0, 1.0], [0.5, 0.0, 1.1], [0.5, 0.5, 1.0]],
        },
    ]


def test_build_task_schema_is_json_serializable() -> None:
    """Verify the generated schema can be serialized as JSON."""
    encoded = json.dumps(llm.task_schema.build_task_schema())

    assert "shapes" in encoded


def test_schema_shapes_match_validation_contracts() -> None:
    """Verify schema shape names come from validation contracts."""
    schema = llm.task_schema.build_task_schema()

    assert tuple(schema["shapes"]) == validation.contracts.SUPPORTED_TRAJECTORY_SHAPES


def test_schema_includes_curriculum_specific_fields_and_shapes() -> None:
    """Verify the LLM schema accepts every current validation field and shape."""
    contracts = validation.contracts
    schema = llm.task_schema.build_task_schema()
    known_fields = set(schema["known_fields"])
    required_by_shape = schema["shape_required_fields"]
    optional_by_shape = schema["shape_optional_fields"]

    assert contracts.SHAPE_START_HOLD_THEN_SHORT_LINE in required_by_shape
    assert contracts.SHAPE_SHORT_SLOW_LINE in required_by_shape
    assert contracts.FIELD_HOLD_DURATION_SEC in required_by_shape[contracts.SHAPE_START_HOLD_THEN_SHORT_LINE]
    assert contracts.FIELD_MOVE_DURATION_SEC in required_by_shape[contracts.SHAPE_START_HOLD_THEN_SHORT_LINE]
    for field in (
        contracts.FIELD_HOLD_DURATION_SEC,
        contracts.FIELD_MOVE_DURATION_SEC,
        contracts.FIELD_RADIUS_X,
        contracts.FIELD_RADIUS_Y,
        contracts.FIELD_START_HOLD_ENABLED,
        contracts.FIELD_START_HOLD_SEC,
        contracts.FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS,
        contracts.FIELD_FINAL_HOLD_ENABLED,
        contracts.FIELD_FINAL_HOLD_SEC,
        contracts.FIELD_EXCLUDE_FINAL_HOLD_FROM_TRACKING_METRICS,
    ):
        assert field in known_fields
    for shape in validation.contracts.SUPPORTED_TRAJECTORY_SHAPES:
        assert shape in required_by_shape
        assert contracts.FIELD_START_HOLD_ENABLED in optional_by_shape[shape]
        assert contracts.FIELD_FINAL_HOLD_ENABLED in optional_by_shape[shape]
        assert contracts.FIELD_FINAL_HOLD_SEC in optional_by_shape[shape]


def test_prompt_contract_includes_supported_shapes_and_json_only_instruction() -> None:
    """Verify prompt contract is bounded to supported JSON task output."""
    prompt_contract = llm.task_schema.build_task_prompt_contract()

    assert "JSON" in prompt_contract
    assert "no prose" in prompt_contract
    assert "python_code" in prompt_contract
    for shape in validation.contracts.SUPPORTED_TRAJECTORY_SHAPES:
        assert shape in prompt_contract


def test_valid_raw_task_normalizes_to_new_dictionary_and_validates() -> None:
    """Verify valid tasks normalize to a copy and pass deterministic validation."""
    raw_task = _valid_tasks_by_shape()[0]

    normalized = llm.task_schema.normalize_proposed_task(raw_task)
    result = llm.task_schema.validate_proposed_task(raw_task)

    assert normalized == raw_task
    assert normalized is not raw_task
    assert result.is_valid


def test_normalization_does_not_mutate_input_mapping() -> None:
    """Verify normalization leaves caller-owned mappings unchanged."""
    raw_task = _valid_tasks_by_shape()[0]
    original = deepcopy(raw_task)

    normalized = llm.task_schema.normalize_proposed_task(raw_task)
    normalized[validation.contracts.FIELD_POSITION] = [1.0, 1.0, 1.0]

    assert raw_task == original


@pytest.mark.parametrize("missing_key", ["task_type", "shape"])
def test_missing_required_top_level_keys_raise_value_error(missing_key: str) -> None:
    """Verify task_type and shape are required before validation."""
    raw_task = _valid_tasks_by_shape()[0]
    raw_task.pop(missing_key)

    with pytest.raises(ValueError, match="missing required keys"):
        llm.task_schema.normalize_proposed_task(raw_task)


@pytest.mark.parametrize("unknown_key", ["python_code", "command", "script", "shell", "imports"])
def test_unsupported_unknown_keys_raise_value_error(unknown_key: str) -> None:
    """Verify unrelated unknown keys are rejected."""
    raw_task = _valid_tasks_by_shape()[0]
    raw_task[unknown_key] = "print('no')"

    with pytest.raises(ValueError, match="unsupported keys"):
        llm.task_schema.normalize_proposed_task(raw_task)


def test_invalid_numeric_values_are_rejected_by_validation_result() -> None:
    """Verify feasibility failures come from deterministic validation."""
    raw_task = _valid_tasks_by_shape()[0]
    raw_task[validation.contracts.FIELD_POSITION] = [3.0, 0.0, 1.0]

    normalized = llm.task_schema.normalize_proposed_task(raw_task)
    result = llm.task_schema.validate_proposed_task(raw_task)

    assert normalized[validation.contracts.FIELD_POSITION] == [3.0, 0.0, 1.0]
    assert not result.is_valid
    assert any("arena" in message for message in result.messages)


def test_all_known_shape_specific_keys_are_accepted_in_valid_tasks() -> None:
    """Verify valid tasks using all supported shape-specific keys pass."""
    for raw_task in _valid_tasks_by_shape():
        normalized = llm.task_schema.normalize_proposed_task(raw_task)
        result = llm.task_schema.validate_proposed_task(normalized)

        assert normalized == raw_task
        assert result.is_valid, result.messages


def test_budget_profile_metadata_is_preserved_but_not_passed_to_validation() -> None:
    """Verify bounded budget profile metadata is schema-only and not part of task validation."""
    raw_task = _valid_tasks_by_shape()[0]
    raw_task[llm.task_schema.STAGE_BUDGET_PROFILE_FIELD] = "recovery"
    raw_task[llm.task_schema.BUDGET_RATIONALE_FIELD] = "Previous stage was unstable but promising."

    normalized = llm.task_schema.normalize_proposed_task(raw_task)
    validation_task = llm.task_schema.task_without_metadata(normalized)
    result = llm.task_schema.validate_proposed_task(raw_task)

    assert normalized[llm.task_schema.STAGE_BUDGET_PROFILE_FIELD] == "recovery"
    assert llm.task_schema.STAGE_BUDGET_PROFILE_FIELD not in validation_task
    assert llm.task_schema.BUDGET_RATIONALE_FIELD not in validation_task
    assert result.is_valid


def test_nested_concrete_task_proposal_is_extracted_and_validated() -> None:
    """Verify wrapper metadata is preserved while the nested concrete task is validated."""
    raw_task = _valid_tasks_by_shape()[0]
    proposal = {
        "proposal_kind": "task",
        "task": raw_task,
        "stage_budget_profile": "recovery",
        "budget_rationale": "Previous stage was unstable but promising.",
    }

    normalized = llm.task_schema.normalize_proposed_task(proposal)
    validation_task = llm.task_schema.task_without_metadata(normalized)
    result = llm.task_schema.validate_proposed_task(proposal)

    assert "task" not in normalized
    assert normalized["task_type"] == raw_task["task_type"]
    assert normalized["shape"] == raw_task["shape"]
    assert normalized[llm.task_schema.STAGE_BUDGET_PROFILE_FIELD] == "recovery"
    assert validation_task == raw_task
    assert result.is_valid


def test_unknown_budget_profile_is_rejected() -> None:
    """Verify arbitrary LLM budget profiles are rejected before validation."""
    raw_task = _valid_tasks_by_shape()[0]
    raw_task[llm.task_schema.STAGE_BUDGET_PROFILE_FIELD] = "full_medium_run"

    with pytest.raises(ValueError, match="stage_budget_profile"):
        llm.task_schema.normalize_proposed_task(raw_task)


def test_raw_timestep_budget_is_rejected() -> None:
    """Verify LLM proposals cannot request arbitrary raw timestep counts."""
    raw_task = _valid_tasks_by_shape()[0]
    raw_task["total_timesteps"] = 999999

    with pytest.raises(ValueError, match="unsupported keys"):
        llm.task_schema.normalize_proposed_task(raw_task)


def test_reason_metadata_is_preserved_but_not_passed_to_validation() -> None:
    """Verify optional reason metadata does not break deterministic validation."""
    raw_task = _valid_tasks_by_shape()[0]
    raw_task[llm.task_schema.REASON_FIELD] = "Next task keeps the drone hovering."

    normalized = llm.task_schema.normalize_proposed_task(raw_task)
    validation_task = llm.task_schema.task_without_metadata(normalized)
    result = llm.task_schema.validate_proposed_task(raw_task)

    assert normalized[llm.task_schema.REASON_FIELD] == "Next task keeps the drone hovering."
    assert llm.task_schema.REASON_FIELD not in validation_task
    assert result.is_valid


@pytest.mark.parametrize(
    ("distribution_id", "expected_path"),
    [
        ("tracking_small", "configs/tasks/task_distribution_tracking_small.yaml"),
        ("hover_bootstrap", "configs/tasks/task_distribution_hover_bootstrap_medium.yaml"),
        ("short_line_bootstrap", "configs/tasks/task_distribution_short_line_bootstrap_medium.yaml"),
        ("vertical_bootstrap", "configs/tasks/task_distribution_vertical_bootstrap_medium.yaml"),
        ("polyline_bootstrap", "configs/tasks/task_distribution_polyline_bootstrap_medium.yaml"),
        ("zigzag_bootstrap", "configs/tasks/task_distribution_zigzag_bootstrap_medium.yaml"),
        ("triangle_bootstrap", "configs/tasks/task_distribution_triangle_bootstrap_medium.yaml"),
        ("multi_height_polyline_bootstrap", "configs/tasks/task_distribution_multi_height_polyline_bootstrap_medium.yaml"),
    ],
)
def test_task_schema_accepts_known_task_distribution_reference(distribution_id: str, expected_path: str) -> None:
    """Verify constrained task-distribution references validate through the schema."""
    proposal = {
        "proposal_kind": "task_distribution",
        "task_distribution_id": distribution_id,
        "reason": "Stay on a known bounded distribution.",
    }

    normalized = llm.task_schema.normalize_proposed_task(proposal)
    result = llm.task_schema.validate_proposed_task(proposal)

    assert normalized["task_distribution_config_path"] == expected_path
    assert result.is_valid


def test_task_schema_rejects_unknown_task_distribution_reference() -> None:
    """Verify LLM proposals cannot invent arbitrary distribution ids."""
    proposal = {"proposal_kind": "task_distribution", "task_distribution_id": "spiral_freeform"}

    with pytest.raises(ValueError, match="task_distribution_id"):
        llm.task_schema.normalize_proposed_task(proposal)


def test_task_schema_accepts_distribution_reference_without_explicit_kind() -> None:
    """Verify distribution-looking proposals do not need concrete task keys."""
    proposal = {
        "task_distribution_id": "tracking_medium",
        "stage_budget_profile": "normal",
        "budget_rationale": "Use the known medium distribution.",
    }

    normalized = llm.task_schema.normalize_proposed_task(proposal)
    result = llm.task_schema.validate_proposed_task(proposal)

    assert normalized["proposal_kind"] == "task_distribution"
    assert normalized["task_distribution_id"] == "tracking_medium"
    assert normalized["task_distribution_config_path"] == "configs/tasks/task_distribution_tracking_medium.yaml"
    assert result.is_valid


def test_prompt_contract_mentions_task_distributions_and_supported_families() -> None:
    """Verify LLM prompts expose constrained task-distribution guidance."""
    prompt_contract = llm.task_schema.build_task_prompt_contract()

    assert "task distribution" in prompt_contract
    assert "tracking_small" in prompt_contract
    assert "tracking_medium" in prompt_contract
    assert "PID" in prompt_contract
    assert "stage_budget_profile" in prompt_contract
    assert "may not propose arbitrary sampling bounds" in prompt_contract
    assert "concrete validated task" in prompt_contract
    assert "bootstrap" in prompt_contract
    assert "short" in prompt_contract
    assert "extend" in prompt_contract
    for family in ("hover_stabilization", "line", "circle", "zigzag", "triangle", "multi_height_polyline"):
        assert family in prompt_contract


def test_task_schema_imports_through_package_alias() -> None:
    """Verify task schema helpers are exposed by the llm package."""
    assert llm.task_schema.build_task_schema is not None


def test_schema_does_not_expose_basic_training_show_as_llm_focused_family() -> None:
    """Verify the Direct-PPO composed show does not become a normal LLM focused family."""
    schema = llm.task_schema.build_task_schema()

    assert "basic_training_show" not in schema["supported_task_distribution_families"]


def test_schema_documents_sampling_bounds_limitation() -> None:
    """Verify Option B is explicit: LLM proposals do not define arbitrary sampling bounds."""
    schema = llm.task_schema.build_task_schema()

    assert "arbitrary sampling-bound proposals are intentionally unsupported" in str(schema["sampling_bounds_scope"])
