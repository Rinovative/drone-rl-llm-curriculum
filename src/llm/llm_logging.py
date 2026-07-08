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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
        encoded = json.dumps(_json_ready(event), sort_keys=True)
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


def _json_ready(value: Any) -> Any:
    """Return a JSON-ready representation of a nested value."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    return value


__all__ = [
    "ProposalEventLogger",
    "read_jsonl",
]
