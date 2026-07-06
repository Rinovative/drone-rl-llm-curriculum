"""Tests for deterministic MVP tracking reward helpers."""

# ruff: noqa: S101

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from src import envs, validation


def _line_task() -> dict[str, object]:
    """Return a valid line task for reward tests."""
    contracts = validation.contracts
    return {
        contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
        contracts.FIELD_SHAPE: contracts.SHAPE_LINE,
        contracts.FIELD_DURATION_SEC: 3.0,
        contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
        contracts.FIELD_START: [0.0, 0.0, 1.0],
        contracts.FIELD_END: [0.5, 0.0, 1.0],
    }


def _reference() -> envs.task_adapter.EnvironmentTaskReference:
    """Return a validated reference for a simple line task."""
    return envs.task_adapter.make_task_reference(_line_task())


def test_zero_error_reward_is_greater_than_larger_error_reward() -> None:
    """Verify larger tracking error lowers the reward."""
    reference_position = np.array([0.0, 0.0, 1.0])

    zero_error_reward = envs.tracking_reward.compute_tracking_reward(
        actual_position=reference_position,
        reference_position=reference_position,
    )
    larger_error_reward = envs.tracking_reward.compute_tracking_reward(
        actual_position=np.array([0.5, 0.0, 1.0]),
        reference_position=reference_position,
    )

    assert zero_error_reward == pytest.approx(0.0)
    assert larger_error_reward < zero_error_reward


def test_action_cost_penalty_reduces_reward() -> None:
    """Verify action magnitude is penalized when supplied."""
    reference_position = np.array([0.0, 0.0, 1.0])
    config = envs.tracking_reward.TrackingRewardConfig(action_cost_weight=0.5)

    no_action_reward = envs.tracking_reward.compute_tracking_reward(reference_position, reference_position, config=config)
    action_reward = envs.tracking_reward.compute_tracking_reward(
        actual_position=reference_position,
        reference_position=reference_position,
        action=np.array([1.0, 2.0]),
        config=config,
    )

    assert action_reward == pytest.approx(-2.5)
    assert action_reward < no_action_reward


def test_reference_sample_selection_covers_first_middle_and_final_steps() -> None:
    """Verify step lookup returns copied reference samples."""
    reference = _reference()
    middle_index = reference.positions.shape[0] // 2
    final_index = reference.positions.shape[0] - 1

    first = envs.tracking_reward.select_reference_position(reference, 0)
    middle = envs.tracking_reward.select_reference_position(reference, middle_index)
    final = envs.tracking_reward.select_reference_position(reference, final_index)

    np.testing.assert_allclose(first, reference.positions[0])
    np.testing.assert_allclose(middle, reference.positions[middle_index])
    np.testing.assert_allclose(final, reference.positions[final_index])
    assert first.flags.owndata


def test_episode_done_at_final_sample_and_configured_max_steps() -> None:
    """Verify terminal flags from trajectory end and max-step cap."""
    reference = _reference()
    final_index = reference.positions.shape[0] - 1

    before_final = envs.tracking_reward.step_tracking_episode(reference, reference.positions[0], 0)
    final = envs.tracking_reward.step_tracking_episode(reference, reference.positions[final_index], final_index)
    capped = envs.tracking_reward.step_tracking_episode(
        reference,
        reference.positions[1],
        1,
        config=envs.tracking_reward.TrackingRewardConfig(max_steps=2),
    )

    assert not before_final.done
    assert final.done
    assert capped.done


def test_reward_helpers_work_with_validated_task_reference() -> None:
    """Verify output package includes copied arrays and expected fields."""
    reference = _reference()

    result = envs.tracking_reward.step_tracking_episode(
        reference=reference,
        actual_position=reference.positions[0],
        step_index=0,
    )

    assert result.step_index == 0
    assert result.position_error_m == pytest.approx(0.0)
    assert result.reward == pytest.approx(0.0)
    assert result.success
    assert result.reference_position.flags.owndata
    assert result.actual_position.flags.owndata


def test_malformed_actual_position_raises_value_error() -> None:
    """Verify actual positions must be finite XYZ vectors."""
    with pytest.raises(ValueError, match="actual_position"):
        envs.tracking_reward.step_tracking_episode(_reference(), [0.0, 1.0], 0)


def test_negative_step_index_raises_value_error() -> None:
    """Verify negative step indices are rejected."""
    with pytest.raises(ValueError, match="nonnegative"):
        envs.tracking_reward.select_reference_position(_reference(), -1)


def test_invalid_reward_config_raises_value_error() -> None:
    """Verify invalid reward configuration is rejected."""
    with pytest.raises(ValueError, match="position_error_weight"):
        envs.tracking_reward.TrackingRewardConfig(position_error_weight=-1.0)
    with pytest.raises(ValueError, match="max_steps"):
        envs.tracking_reward.TrackingRewardConfig(max_steps=0)


def test_empty_reference_raises_value_error() -> None:
    """Verify empty reference arrays are rejected."""
    reference = envs.task_adapter.EnvironmentTaskReference(
        task=MappingProxyType({"shape": "line"}),
        shape="line",
        times=np.array([], dtype=float),
        positions=np.empty((0, 3), dtype=float),
        validation_messages=(),
    )

    with pytest.raises(ValueError, match="at least one sample"):
        envs.tracking_reward.select_reference_position(reference, 0)
