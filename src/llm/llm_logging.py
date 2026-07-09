"""
===============================================================================
llm_logging.py
===============================================================================
Write JSONL proposal events for LLM-guided curriculum runs.

Responsibilities:
  - Append deterministic JSON objects to a proposal event log
  - Create run-scoped LLM log directories only when explicitly used
  - Convert pathlib and tuple values into JSON-ready structures

Design principles:
  - Keep logs line-oriented and easy to inspect
  - Avoid adding timestamps so smoke tests remain deterministic

Boundaries:
  - Artifact path resolution belongs in utils_artifacts.py
  - Proposal generation and validation status belong in llm_curriculum.py
===============================================================================

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src import utils

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class ProposalEventLogger:
    """
    Append proposal events to a JSONL file.

    Parameters
    ----------
    log_path
        Destination JSONL path, usually ``storage/runs/<run>/llm_logs/proposals.jsonl``.

    """

    log_path: Path

    def append(self, event: Mapping[str, Any]) -> None:
        """
        Append one JSON-serializable event to the log.

        Parameters
        ----------
        event
            Event mapping produced by the proposal pipeline.

        """
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        safe_event = utils.serialization.to_jsonable(event)
        utils.serialization.assert_json_serializable(safe_event, "LLM proposal event")
        encoded = json.dumps(safe_event, sort_keys=True, allow_nan=False)
        with self.log_path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(f"{encoded}\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """
    Read a JSONL proposal log into memory.

    Parameters
    ----------
    path
        Proposal log path.

    Returns
    -------
    list[dict[str, Any]]
        Decoded log events.

    """
    log_path = Path(path)
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]


__all__ = [
    "ProposalEventLogger",
    "read_jsonl",
]
