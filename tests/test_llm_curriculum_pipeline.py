"""Tests for LLM proposal parsing, repair, validation, and event logging."""

# ruff: noqa: S101

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from src import llm

VALID_TASK_JSON = (
    '{"task_type":"trajectory","shape":"hover_stabilization","duration_sec":2.0,'
    '"sample_rate_hz":10.0,"position":[0.0,0.0,1.0],"reason":"Keep the first task stable."}'
)
REPAIRED_TASK_JSON = (
    '{"task_type":"trajectory","shape":"nearby_target_hover","duration_sec":2.5,'
    '"sample_rate_hz":10.0,"position":[0.1,0.0,1.0],"reason":"Small target offset after hover."}'
)
REPAIRED_PROPOSAL_COUNT = 2

INVALID_VALIDATION_JSON = '{"task_type":"trajectory","shape":"hover_stabilization","duration_sec":2.0,"sample_rate_hz":10.0,"position":[3.0,0.0,1.0]}'


def _context() -> llm.curriculum.ProposalContext:
    """Return a minimal valid proposal context."""
    return llm.curriculum.ProposalContext(curriculum_name="curriculum_llm_test", stage_index=2)


def _logger(path: Path) -> llm.logging.ProposalEventLogger:
    """Return a proposal logger for tests."""
    return llm.logging.ProposalEventLogger(path / "proposals.jsonl")


def test_valid_mock_proposal_is_accepted_and_logged(tmp_path: Path) -> None:
    """Verify a valid JSON task is accepted without repair."""
    logger = _logger(tmp_path)
    result = llm.curriculum.propose_next_task(
        client=llm.client.MockLLMClient([VALID_TASK_JSON]),
        context=_context(),
        settings=llm.curriculum.ProposalSettings(max_repair_attempts=1),
        logger=logger,
    )
    events = llm.logging.read_jsonl(logger.log_path)

    assert result.task is not None
    assert result.task["shape"] == "hover_stabilization"
    assert "reason" not in result.task
    assert result.task_reason == "Keep the first task stable."
    assert result.stats["total_proposals"] == 1
    assert result.stats["invalid_proposals"] == 0
    assert events[0]["status"] == "accepted"
    assert events[0]["validation_status"] == "valid"


def test_nested_concrete_task_proposal_is_accepted_and_logged(tmp_path: Path) -> None:
    """Verify nested task wrappers are accepted without top-level task_type or shape."""
    logger = _logger(tmp_path)
    response = (
        '{"proposal_kind":"task","task":{"task_type":"trajectory","shape":"hover_stabilization",'
        '"duration_sec":2.0,"sample_rate_hz":10.0,"position":[0.0,0.0,1.0]},'
        '"stage_budget_profile":"normal","budget_rationale":"Normal verification budget."}'
    )

    result = llm.curriculum.propose_next_task(
        client=llm.client.MockLLMClient([response]),
        context=_context(),
        settings=llm.curriculum.ProposalSettings(max_repair_attempts=0),
        logger=logger,
    )
    events = llm.logging.read_jsonl(logger.log_path)

    assert result.task is not None
    assert result.proposal_type == "task"
    assert result.original_proposal is not None
    assert "task" in result.original_proposal
    assert result.normalized_proposal is not None
    assert "task" not in result.normalized_proposal
    assert result.task["shape"] == "hover_stabilization"
    assert result.stage_budget_profile == "normal"
    assert events[0]["original_proposal"]["task"]["shape"] == "hover_stabilization"
    assert events[0]["accepted_task"]["shape"] == "hover_stabilization"
    assert events[0]["resolved_task_shape"] == "hover_stabilization"


def test_invalid_proposal_is_repaired_once_and_accepted(tmp_path: Path) -> None:
    """Verify parse failures trigger one bounded repair prompt and success accounting."""
    logger = _logger(tmp_path)
    result = llm.curriculum.propose_next_task(
        client=llm.client.MockLLMClient(["Here is a task", REPAIRED_TASK_JSON]),
        context=_context(),
        settings=llm.curriculum.ProposalSettings(max_repair_attempts=1),
        logger=logger,
    )
    events = llm.logging.read_jsonl(logger.log_path)

    assert result.task is not None
    assert result.task["shape"] == "nearby_target_hover"
    assert result.stats["total_proposals"] == REPAIRED_PROPOSAL_COUNT
    assert result.stats["invalid_proposals"] == 1
    assert result.stats["repair_attempts"] == 1
    assert result.stats["repair_successes"] == 1
    assert events[0]["status"] == "rejected"
    assert events[0]["error_type"] == "parse"
    assert events[1]["status"] == "accepted"
    assert events[1]["is_repair_attempt"] is True


def test_invalid_proposal_fails_after_max_repair_attempts(tmp_path: Path) -> None:
    """Verify proposal failure is clear after exhausting repair attempts."""
    logger = _logger(tmp_path)

    with pytest.raises(llm.curriculum.LLMCurriculumProposalError, match="failed after 2 attempt"):
        llm.curriculum.propose_next_task(
            client=llm.client.MockLLMClient(["not json", "still not json"]),
            context=_context(),
            settings=llm.curriculum.ProposalSettings(max_repair_attempts=1),
            logger=logger,
        )

    events = llm.logging.read_jsonl(logger.log_path)
    assert [event["status"] for event in events] == ["rejected", "rejected"]
    assert all(event["error_type"] == "parse" for event in events)


def test_event_log_contains_validation_status_and_rejection_reasons(tmp_path: Path) -> None:
    """Verify validation failures are logged with concrete rejection reasons."""
    logger = _logger(tmp_path)
    result = llm.curriculum.propose_next_task(
        client=llm.client.MockLLMClient([INVALID_VALIDATION_JSON]),
        context=_context(),
        settings=llm.curriculum.ProposalSettings(max_repair_attempts=0, skip_invalid_proposals=True),
        logger=logger,
    )
    events = llm.logging.read_jsonl(logger.log_path)

    assert result.task is None
    assert result.stats["invalid_proposals"] == 1
    assert events[0]["status"] == "rejected"
    assert events[0]["error_type"] == "validation"
    assert events[0]["validation_status"] == "invalid"
    assert any("arena" in reason for reason in events[0]["rejection_reasons"])


def test_duplicate_consecutive_task_family_is_rejected_and_repaired(tmp_path: Path) -> None:
    """Verify immediate duplicate task families are rejected before accepting a repair."""
    logger = _logger(tmp_path)
    duplicate_hover = (
        '{"task_type":"trajectory","shape":"nearby_target_hover","duration_sec":2.5,'
        '"sample_rate_hz":10.0,"position":[0.1,0.0,1.0],"reason":"Repeat hover."}'
    )
    repaired_line = (
        '{"task_type":"trajectory","shape":"line","duration_sec":3.0,'
        '"sample_rate_hz":10.0,"start":[0.0,0.0,1.0],"end":[0.35,0.0,1.0],"reason":"Switch to line."}'
    )
    context = llm.curriculum.ProposalContext(
        curriculum_name="curriculum_llm_test",
        stage_index=2,
        recent_accepted_tasks=({"accepted_stage_task_shape": "hover_stabilization"},),
    )

    result = llm.curriculum.propose_next_task(
        client=llm.client.MockLLMClient([duplicate_hover, repaired_line]),
        context=context,
        settings=llm.curriculum.ProposalSettings(max_repair_attempts=1),
        logger=logger,
    )
    events = llm.logging.read_jsonl(logger.log_path)

    assert result.task is not None
    assert result.task["shape"] == "line"
    assert result.stats["duplicate_task_rejections"] == 1
    assert result.stats["repair_successes"] == 1
    assert events[0]["status"] == "rejected"
    assert events[0]["error_type"] == "duplicate_task"
    assert events[0]["validation_status"] == "duplicate"
    assert events[0]["previous_stage_task_shape"] == "hover_stabilization"
    assert events[0]["requested_stage_task_shape"] == "nearby_target_hover"
    assert events[0]["duplicate_task_rejected"] is True
    assert events[1]["status"] == "accepted"
    assert events[1]["accepted_stage_task_shape"] == "line"


def test_valid_task_distribution_proposal_is_accepted_and_logged(tmp_path: Path) -> None:
    """Verify a constrained distribution reference proposal is accepted and logged."""
    logger = _logger(tmp_path)
    response = '{"proposal_kind":"task_distribution","task_distribution_id":"tracking_small","reason":"Stay conservative."}'

    result = llm.curriculum.propose_next_task(
        client=llm.client.MockLLMClient([response]),
        context=_context(),
        settings=llm.curriculum.ProposalSettings(max_repair_attempts=0),
        logger=logger,
    )
    events = llm.logging.read_jsonl(logger.log_path)

    assert result.task is not None
    assert result.proposal_type == "task_distribution"
    assert result.normalized_proposal is not None
    assert result.task["proposal_kind"] == "task_distribution"
    assert result.task["task_distribution_id"] == "tracking_small"
    assert events[0]["proposal_type"] == "task_distribution"
    assert events[0]["task_distribution_reference"]["task_distribution_id"] == "tracking_small"
    assert events[0]["accepted_task"]["task_distribution_config_path"] == "configs/tasks/task_distribution_tracking_small.yaml"


def test_distribution_reference_without_kind_is_accepted_and_logged(tmp_path: Path) -> None:
    """Verify distribution references may omit proposal_kind without becoming concrete tasks."""
    logger = _logger(tmp_path)
    response = '{"task_distribution_config_path":"configs/tasks/task_distribution_tracking_medium.yaml","reason":"Use medium distribution."}'

    result = llm.curriculum.propose_next_task(
        client=llm.client.MockLLMClient([response]),
        context=_context(),
        settings=llm.curriculum.ProposalSettings(max_repair_attempts=0),
        logger=logger,
    )
    events = llm.logging.read_jsonl(logger.log_path)

    assert result.task is not None
    assert result.proposal_type == "task_distribution"
    assert result.task["task_distribution_id"] == "tracking_medium"
    assert events[0]["status"] == "accepted"
    assert events[0]["proposal_type"] == "task_distribution"
    assert events[0]["task_distribution_reference"]["task_distribution_id"] == "tracking_medium"


def test_repair_prompt_mentions_supported_distributions_and_concrete_task_values() -> None:
    """Verify repair prompts include concrete task-distribution repair guidance."""
    messages = llm.prompts.build_task_repair_messages(
        curriculum_name="curriculum_llm_test",
        stage_index=2,
        recent_accepted_tasks=(),
        recent_rejected_tasks=(),
        metrics_summary={"failure_primary_mode": "reference_too_fast", "status": "needs_easier_task"},
        curriculum_history=({"stage_index": 1, "accepted_task_family": "hover_stabilization"},),
        curriculum_summary={"position_error_trend": "worsening"},
        recent_context_limit=3,
        previous_response="{}",
        error_messages=("unsupported family",),
    )
    content = messages[-1]["content"]

    assert "supported" in content
    assert "concrete safe task values" in content
    assert "known distribution ids/paths" in content
    assert "tracking_small" in content
    assert "curriculum_history" in content
    assert "readiness_level_omitted" in content
    assert "reference_too_fast" in content
    assert "z_instability" in content
    assert "curriculum_feedback" in content
    assert "controlled vertical" in content
    assert "slow L-shape/polyline" in content
    assert "gentle ellipse/circle" in content
    assert "Do not choose broad shows, scenarios, or basic_training_show" in content
    assert "true instability" in content


def test_proposal_prompt_embeds_structured_curriculum_feedback_guidance() -> None:
    """Verify proposal prompts include compact structured feedback and constructive guidance."""
    messages = llm.prompts.build_task_proposal_messages(
        curriculum_name="curriculum_llm_test",
        stage_index=3,
        recent_accepted_tasks=({"stage_index": 2, "accepted_task_family": "line", "task_shape": "line"},),
        recent_rejected_tasks=(),
        metrics_summary={
            "failure_primary_mode": "z_instability",
            "curriculum_feedback_summary": "Altitude control weak; use controlled altitude practice.",
            "curriculum_primary_skill_gaps": ["altitude_control"],
            "curriculum_recommended_next_task_families": [
                {
                    "task_family": "takeoff_stabilization",
                    "reason": "controlled z practice",
                    "targeted_skill": "altitude_control",
                    "difficulty_hint": "low",
                    "priority": 1,
                }
            ],
        },
        curriculum_history=({"stage_index": 2, "feedback_summary": {"primary_skill_gaps": ["altitude_control"]}},),
        curriculum_summary={"previous_feedback_summaries": [{"primary_skill_gaps": ["altitude_control"]}]},
        recent_context_limit=3,
    )
    content = messages[-1]["content"]

    assert "curriculum_feedback" in content
    assert "guidance_not_absolute_command" in content
    assert "takeoff_stabilization" in content
    assert "controlled vertical" in content
    assert "shorter or slower line" in content
    assert "slow polyline or L-shape" in content
    assert "gentle ellipse or slow circle" in content
    assert "easier or slower same-family variant" in content


def test_invalid_budget_profile_is_repaired_once_and_accepted(tmp_path: Path) -> None:
    """Verify invalid budget profile metadata is repairable through the normal loop."""
    logger = _logger(tmp_path)
    invalid_budget_json = (
        '{"task_type":"trajectory","shape":"hover_stabilization","duration_sec":2.0,'
        '"sample_rate_hz":10.0,"position":[0.0,0.0,1.0],"stage_budget_profile":"full_medium"}'
    )
    repaired_budget_json = (
        '{"task_type":"trajectory","shape":"hover_stabilization","duration_sec":2.0,'
        '"sample_rate_hz":10.0,"position":[0.0,0.0,1.0],"stage_budget_profile":"normal",'
        '"budget_rationale":"Normal progression after repair."}'
    )

    result = llm.curriculum.propose_next_task(
        client=llm.client.MockLLMClient([invalid_budget_json, repaired_budget_json]),
        context=_context(),
        settings=llm.curriculum.ProposalSettings(max_repair_attempts=1),
        logger=logger,
    )
    events = llm.logging.read_jsonl(logger.log_path)

    assert result.task is not None
    assert result.stage_budget_profile == "normal"
    assert result.budget_rationale == "Normal progression after repair."
    assert result.stats["invalid_proposals"] == 1
    assert events[0]["error_type"] == "schema"
    assert any("stage_budget_profile" in reason for reason in events[0]["rejection_reasons"])
    assert events[1]["stage_budget_profile"] == "normal"
