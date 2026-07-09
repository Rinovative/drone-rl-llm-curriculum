"""Tests for LLM proposal parsing, repair, validation, and event logging."""

# ruff: noqa: S101

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
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
LOW_TEST_PROMPT_CONTEXT_LIMIT = 4096
LOW_TEST_PROMPT_BUDGET = 900
CONTEXT_RETRY_SUCCESS_CALL_COUNT = 2
CONTEXT_RETRY_FAILURE_CALL_COUNT = 3
CONTEXT_RETRY_FAILURE_COUNT = 2

INVALID_VALIDATION_JSON = '{"task_type":"trajectory","shape":"hover_stabilization","duration_sec":2.0,"sample_rate_hz":10.0,"position":[3.0,0.0,1.0]}'


class ContextOverflowClient:
    """Mock client that raises provider context-size errors before succeeding."""

    def __init__(self, *, failures_before_success: int, response: str = VALID_TASK_JSON) -> None:
        """Initialize the client with a fixed number of context failures."""
        self.failures_before_success = failures_before_success
        self.response = response
        self.messages: list[Sequence[Mapping[str, str]]] = []

    def complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Raise context overflow until the configured failure count is exhausted."""
        self.messages.append(messages)
        if len(self.messages) <= self.failures_before_success:
            message = "OpenAI-compatible LLM request failed with HTTP 400: request (16745 tokens) exceeds the available context size (16384 tokens)"
            raise llm.client.LLMClientError(message)
        return self.response


def _prompt_context(messages: Sequence[Mapping[str, str]], *, label: str = "Context JSON") -> dict[str, object]:
    """Extract the embedded prompt context JSON from messages."""
    content = str(messages[-1]["content"])
    payload = content.split(f"{label}:\n", maxsplit=1)[1]
    parsed = json.loads(payload)
    assert isinstance(parsed, dict)
    return parsed


def _history_item(stage_index: int) -> dict[str, object]:
    """Return a noisy stage-history item that prompt compaction must summarize."""
    return {
        "stage_index": stage_index,
        "stage_name": f"stage_{stage_index}",
        "accepted_task_family": "line" if stage_index % 2 else "vertical_up_down",
        "task_shape": "line",
        "task_distribution_id": "short_line_bootstrap",
        "selected_stage_budget_profile": "normal",
        "stage_total_timesteps": 250000,
        "accepted_task": {"sentinel": "FULL_ACCEPTED_TASK_SENTINEL", "payload": [stage_index] * 64},
        "resolved_task": {"sentinel": "FULL_RESOLVED_TASK_SENTINEL", "payload": [stage_index] * 64},
        "resolved_task_sample_metadata": {
            "task_distribution_sampled_family": "line",
            "task_distribution_sampled_task": {"sentinel": "FULL_SAMPLE_METADATA_SENTINEL"},
        },
        "metrics": {"mean_position_error_tracking_m": 0.2 + stage_index / 100.0, "failure_primary_mode": "z_instability"},
        "feedback_summary": {
            "llm_instruction_summary": "Use compact altitude feedback.",
            "primary_skill_gaps": ["altitude_control"],
            "diagnostic_signals": {"z_instability": {"evidence": ["long evidence"] * 10}},
        },
    }


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


def test_full_stage_three_prompt_estimate_stays_under_default_budget() -> None:
    """Verify a stage-3 prompt with two previous stages stays below the safe budget."""
    history = tuple(_history_item(index) for index in range(1, 3))
    messages = llm.prompts.build_task_proposal_messages(
        curriculum_name="curriculum_llm_test",
        stage_index=3,
        recent_accepted_tasks=history,
        recent_rejected_tasks=(),
        metrics_summary={"curriculum_primary_skill_gaps": ["altitude_control"]},
        curriculum_history=history,
        curriculum_summary={"accepted_task_family_counts": {"line": 1, "vertical_up_down": 1}, "trend_status": "flat"},
        recent_context_limit=3,
    )

    assert llm.prompts.estimate_messages_tokens(messages) < llm.prompts.DEFAULT_PROMPT_BUDGET_TOKENS
    context = _prompt_context(messages)
    assert context["compaction_mode"] == "full"
    assert "task_catalog" in context


def test_compact_prompt_keeps_identities_and_drops_full_payloads() -> None:
    """Verify compact prompts keep useful history without full task/sample payloads."""
    history = tuple(_history_item(index) for index in range(1, 11))
    messages = llm.prompts.build_task_proposal_messages(
        curriculum_name="curriculum_llm_test",
        stage_index=11,
        recent_accepted_tasks=history[-3:],
        recent_rejected_tasks=(
            {
                "stage_index": 10,
                "response_text": "FULL_RAW_RESPONSE_SENTINEL" * 100,
                "parsed_task": {"sentinel": "FULL_REJECTED_TASK_SENTINEL"},
                "error_type": "validation",
                "rejection_reasons": ["too fast", "too high", "too long", "too wide"],
            },
        ),
        metrics_summary={
            "curriculum_feedback_summary": "Altitude control remains weak.",
            "curriculum_primary_skill_gaps": ["altitude_control", "xy_tracking", "turn_following", "extra"],
            "curriculum_recommended_next_task_families": [
                {"task_family": "vertical_up_down", "targeted_skill": "altitude_control", "difficulty_hint": "low", "priority": 1}
            ],
            "curriculum_strategy": {"candidate_next_skills": ["altitude_control", "xy_tracking", "turn_following", "curvature"]},
        },
        curriculum_history=history,
        curriculum_summary={
            "accepted_task_family_counts": {"line": 5, "vertical_up_down": 5},
            "last_accepted_task_family": "vertical_up_down",
            "trend_status": "worsening",
        },
        recent_context_limit=3,
        compaction_mode=llm.prompts.PROMPT_COMPACTION_COMPACT,
    )
    content = messages[-1]["content"]
    context = _prompt_context(messages)

    assert llm.prompts.estimate_messages_tokens(messages) < llm.prompts.DEFAULT_PROMPT_BUDGET_TOKENS
    assert context["compaction_mode"] == "compact"
    assert "short_line_bootstrap" in content
    assert "vertical_up_down_bootstrap" in content
    assert "FULL_ACCEPTED_TASK_SENTINEL" not in content
    assert "FULL_RESOLVED_TASK_SENTINEL" not in content
    assert "FULL_SAMPLE_METADATA_SENTINEL" not in content
    assert "FULL_RAW_RESPONSE_SENTINEL" not in content
    assert context["curriculum_history"][-1]["accepted_distribution_id"] == "short_line_bootstrap"
    assert context["curriculum_feedback"]["primary_skill_gaps"] == ["altitude_control", "xy_tracking", "turn_following"]


def test_oversized_prompt_uses_compaction_before_llm_call() -> None:
    """Verify the budget layer moves oversized context to compact or minimal mode."""
    huge_history = tuple(_history_item(index) for index in range(1, 15))
    context = llm.curriculum.ProposalContext(
        curriculum_name="curriculum_llm_test",
        stage_index=15,
        recent_accepted_tasks=huge_history[-3:],
        curriculum_history=huge_history,
        metrics_summary={"curriculum_primary_skill_gaps": ["altitude_control"]},
        curriculum_summary={"accepted_task_family_counts": {"line": 7, "vertical_up_down": 7}},
    )
    settings = llm.curriculum.ProposalSettings(
        max_repair_attempts=0,
        prompt_context_limit_tokens=LOW_TEST_PROMPT_CONTEXT_LIMIT,
        prompt_response_reserve_tokens=1200,
        prompt_budget_tokens=LOW_TEST_PROMPT_BUDGET,
    )

    _messages, metadata = llm.curriculum._budgeted_messages_for_attempt(  # noqa: SLF001
        context=context,
        settings=settings,
        attempt_index=0,
        previous_response="",
        previous_errors=(),
        initial_mode=llm.prompts.PROMPT_COMPACTION_FULL,
    )

    assert metadata["llm_prompt_context_limit"] == LOW_TEST_PROMPT_CONTEXT_LIMIT
    assert metadata["llm_prompt_budget_tokens"] == LOW_TEST_PROMPT_BUDGET
    assert metadata["llm_prompt_compaction_mode"] in {"compact", "minimal"}
    assert "full_accepted_task_json" in metadata["llm_prompt_sections_dropped"]


def test_context_overflow_retries_compact_prompt_and_uses_success(tmp_path: Path) -> None:
    """Verify provider context overflow retries with a smaller prompt before accepting."""
    logger = _logger(tmp_path)
    client = ContextOverflowClient(failures_before_success=1)

    result = llm.curriculum.propose_next_task(
        client=client,
        context=_context(),
        settings=llm.curriculum.ProposalSettings(max_repair_attempts=0),
        logger=logger,
    )
    events = llm.logging.read_jsonl(logger.log_path)

    assert result.task is not None
    assert len(client.messages) == CONTEXT_RETRY_SUCCESS_CALL_COUNT
    assert result.stats["llm_request_failed_due_to_context_size"] is True
    assert result.stats["llm_context_retry_count"] == 1
    assert result.stats["llm_context_retry_modes"] == ["compact"]
    assert result.stats["llm_context_fallback_used"] is False
    assert events[0]["llm_prompt_compaction_mode"] == "compact"
    assert events[0]["llm_prompt_estimated_tokens"] <= events[0]["llm_prompt_budget_tokens"]


def test_context_overflow_after_minimal_prompt_raises_proposal_error() -> None:
    """Verify repeated context overflow is converted to proposal metadata for fallback."""
    client = ContextOverflowClient(failures_before_success=3)

    with pytest.raises(llm.curriculum.LLMCurriculumProposalError, match="context size") as exc_info:
        llm.curriculum.propose_next_task(
            client=client,
            context=_context(),
            settings=llm.curriculum.ProposalSettings(max_repair_attempts=0),
        )

    stats = exc_info.value.stats
    assert len(client.messages) == CONTEXT_RETRY_FAILURE_CALL_COUNT
    assert stats["llm_request_failed_due_to_context_size"] is True
    assert stats["llm_context_retry_count"] == CONTEXT_RETRY_FAILURE_COUNT
    assert stats["llm_context_retry_modes"] == ["compact", "minimal"]
    assert stats["llm_context_fallback_used"] is True
    assert "exceeds the available context size" in stats["llm_context_fallback_reason"]
