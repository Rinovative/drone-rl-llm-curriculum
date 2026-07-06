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
        },
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_LINE,
            contracts.FIELD_DURATION_SEC: 3.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_START: [0.0, 0.0, 1.0],
            contracts.FIELD_END: [0.5, 0.0, 1.0],
        },
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_VERTICAL,
            contracts.FIELD_DURATION_SEC: 3.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_XY: [0.25, -0.25],
            contracts.FIELD_START_HEIGHT: 0.8,
            contracts.FIELD_END_HEIGHT: 1.4,
        },
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_POLYLINE,
            contracts.FIELD_DURATION_SEC: 6.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
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


def test_prompt_contract_includes_supported_shapes_and_json_only_instruction() -> None:
    """Verify prompt contract is bounded to supported JSON task output."""
    prompt_contract = llm.task_schema.build_task_prompt_contract()

    assert "JSON" in prompt_contract
    assert "no prose" in prompt_contract
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


@pytest.mark.parametrize("unknown_key", ["python_code", "command"])
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


def test_reason_metadata_is_preserved_but_not_passed_to_validation() -> None:
    """Verify optional reason metadata does not break deterministic validation."""
    raw_task = _valid_tasks_by_shape()[0]
    raw_task[llm.task_schema.REASON_FIELD] = "Next task keeps the drone hovering."

    normalized = llm.task_schema.normalize_proposed_task(raw_task)
    result = llm.task_schema.validate_proposed_task(raw_task)

    assert normalized[llm.task_schema.REASON_FIELD] == "Next task keeps the drone hovering."
    assert result.is_valid


def test_task_schema_imports_through_package_alias() -> None:
    """Verify task schema helpers are exposed by the llm package."""
    assert llm.task_schema.build_task_schema is not None
