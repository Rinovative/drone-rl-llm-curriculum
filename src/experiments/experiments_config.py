"""
===============================================================================
experiments_config.py
===============================================================================
Load experiment configuration files for shared project entry points.

Responsibilities:
  - Load YAML experiment configuration files from explicit paths
  - Return plain dictionaries for training, evaluation, validation, and LLM callers

Design principles:
  - Keep loading deterministic and side-effect free
  - Avoid schema decisions until downstream experiment orchestration needs them

Boundaries:
  - Task feasibility checks belong in validation modules
  - Training, simulation, and artifact creation belong in dedicated experiment modules
===============================================================================

"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """
    Load an experiment configuration from a YAML file.

    Parameters
    ----------
    path
        Path to a YAML experiment configuration file.

    Returns
    -------
    dict[str, Any]
        Plain dictionary containing the loaded experiment configuration.

    Raises
    ------
    FileNotFoundError
        If the configuration path does not exist.
    ValueError
        If the YAML file is empty or its root is not a mapping.

    """
    config_path = Path(path)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if loaded is None:
        message = "experiment config must not be empty"
        raise ValueError(message)
    if not isinstance(loaded, Mapping):
        message = "experiment config root must be a mapping"
        raise ValueError(message)  # noqa: TRY004 - public contract requires ValueError.

    return cast("dict[str, Any]", dict(loaded))


__all__ = [
    "load_experiment_config",
]
