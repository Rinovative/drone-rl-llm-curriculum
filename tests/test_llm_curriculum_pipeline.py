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
