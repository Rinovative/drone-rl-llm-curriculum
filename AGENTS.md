# Agent Instructions

Before making changes, read `PROJECT_BRIEF.md`.

This repository is a research project on drone reinforcement learning with LLM-guided curriculum generation.

## Codex Environment Notes

This repository is used inside a Docker development container through the VS Code Codex extension.

Docker is the outer isolation boundary.

Codex should use:

```text
sandbox_mode = "danger-full-access"
approval_policy = "on-request"
```

Normal sandboxed shell execution is known to be unreliable in this environment.

The sandbox wrapper may fail before shell commands start with `bwrap`, namespace, or sandbox-wrapper errors. This is a known environment limitation, not a project error.

Codex should therefore use the approval/escalation path directly for shell commands in this repository instead of first trying the normal sandboxed shell path.

Rules for shell commands:

- Use scoped escalation from the start for shell commands.
- Keep escalated commands narrow, task-relevant, and easy to review.
- Do not use escalation as permission to broaden the task.
- Do not run destructive commands unless the user explicitly requested them.
- Do not repeatedly diagnose or retry sandbox-wrapper, `bwrap`, or namespace failures.
- If a startup-level sandbox failure still appears, switch immediately to the fallback path described below.
- For read-only inspection commands, scoped escalation is allowed.
- For mutating commands, scoped escalation is allowed only when the command is required for the approved task.

Examples of acceptable scoped shell commands:

```bash
git status --short
git diff -- path/to/file
sed -n '1,220p' path/to/file
ruff format .
ruff check . --fix
mypy src
pytest -q
python -m json.tool path/to/file.json >/dev/null
```

Do not change Docker setup, training behavior, notebook behavior, task queues, or project logic unless the user explicitly asks for that.

## Editing Workflow

Before editing, inspect the current file content and the current diff.

Recommended inspection commands:

```bash
git status --short
git diff -- path/to/file
sed -n '1,220p' path/to/file
```

Use small, targeted edits.

In this repository, do not use `apply_patch` as the default edit path for existing repository files.

Normal sandboxed shell execution and patch helpers are known to be unreliable in this VS Code Codex container environment. For existing repository files, use targeted Python `pathlib` read/modify/write directly.

Preferred edit order:

1. Inspect the current file content and current diff.
2. For existing repository files, use a targeted Python `pathlib` edit.
3. Use exact replacements whenever possible.
4. If a full-file replacement is explicitly requested by the user, write the full file deliberately with Python `pathlib`.
5. After editing, inspect the resulting diff.
6. Run the smallest relevant validation command.

`apply_patch` may only be used when:

- The user explicitly asks for patch-style output, or
- The edit is first tested harmlessly outside the repository, for example in `/tmp` or `/workspace/storage/tmp`, and the command path is known to work.

Do not repeatedly try `apply_patch` after any sandbox-wrapper, `bwrap`, namespace, helper, or patch-context failure.

Do not use fragile shell editing commands such as long `sed -i` chains, broad regex replacements, or unreviewed multiline `echo` commands for source files.

When using Python `pathlib`, preserve UTF-8 text encoding and newline formatting:

```python
from pathlib import Path

path = Path("path/to/file.py")
text = path.read_text(encoding="utf-8")
text = text.replace("old exact text", "new exact text")
path.write_text(text, encoding="utf-8")
```

After editing, always inspect the resulting diff:

```bash
git diff -- path/to/file
```

For structured files, validate immediately after editing:

```bash
python -m json.tool path/to/file.json >/dev/null
python -m tomllib path/to/file.toml >/dev/null
python - <<'PY'
from pathlib import Path
import yaml

yaml.safe_load(Path("path/to/file.yml").read_text(encoding="utf-8"))
PY
```

If `yaml` is unavailable, validate YAML through the smallest available project command that loads the file.

Do not modify unrelated files to make a patch easier.

Do not reformat the entire repository unless the user explicitly asks for full formatting or the task includes repository-wide formatting.

When a command modifies files automatically, inspect the diff before continuing.

If a tool or command creates unexpected changes, stop and report them.

## Commit Policy

Agents must never create commits.

The user creates all commits manually.

Agents may edit files, run formatters, run linters, run type checks, and run tests when requested or when useful for validation.

Before handing work back to the user, agents should run the relevant quality checks and fix formatting or linting issues directly when the fix is safe and local.

Recommended checks:

```bash
ruff check .
ruff format --check .
mypy src
```

If formatting changes are needed, run:

```bash
ruff format .
ruff check . --fix
```

Do not run long training jobs unless explicitly requested.

## Plan Mode Rules

In Plan Mode:

- Do not edit files.
- Do not run mutating commands.
- Do not create commits.
- Read-only inspection commands are allowed when needed.
- Use scoped escalation for read-only shell commands according to the Codex Environment Notes.
- Return a concrete implementation plan before making changes.

A good Plan Mode response should include:

1. Files to create or modify.
2. Public API to expose.
3. Tests to add or update.
4. Validation commands.
5. Risks, uncertainties, or conflicts with existing code.
6. A step-by-step implementation order.

## Goal Mode Rules

In Goal Mode:

- Implement only the approved task.
- Do not create commits.
- Do not edit unrelated files.
- Follow `AGENTS.md` exactly.
- Run the smallest useful validation during the task.
- Run the requested final validation commands before handing work back.
- Report changed files, validation results, and any remaining issues.
- Provide a short manual notebook check cell when the task affects user-facing behavior.

## Architecture Rules

Core project logic belongs in `src/`.

Shell, Docker, HPC, and queue helper scripts belong in `scripts/`.

Experiment configuration files belong in `configs/`.

Curated figures and documentation media belong in `docs/`.

Large generated artifacts, models, videos, logs, W&B data, generated datasets, and temporary files belong in `storage/`.

Do not commit generated training artifacts.

The final notebook should explain and demonstrate the project, but reusable logic belongs in `src/`.

## Import Style

Use the project import style:

```python
from src import envs, evaluation, experiments, llm, trajectories, utils, validation
```

Use package aliases consistently in code.

Preferred style:

```python
from src import envs, trajectories, validation

env = envs.builders.make_tracking_env(config)
path = trajectories.primitives.make_circle_trajectory(spec)
result = validation.tasks.validate_task(task)
```

Avoid importing deep implementation modules directly when a package alias exists.

Avoid this style unless there is a concrete reason:

```python
from src.envs.envs_builders import make_tracking_env
from src.trajectories.trajectories_primitives import make_circle_trajectory
from src.validation.validation_tasks import validate_task
```

Do not introduce a separate top-level package such as `drone_curriculum`.

## File Naming Style in `src`

Python files inside a source package must use the package-prefix naming style.

Pattern:

```text
src/<package>/<package>_<responsibility>.py
```

Examples:

```text
src/envs/envs_builders.py
src/envs/envs_rewards.py
src/envs/envs_observations.py
src/trajectories/trajectories_primitives.py
src/trajectories/trajectories_formations.py
src/llm/llm_prompts.py
src/llm/llm_curriculum.py
src/validation/validation_tasks.py
src/validation/validation_contracts.py
src/evaluation/evaluation_metrics.py
src/evaluation/evaluation_plots.py
src/utils/utils_paths.py
src/utils/utils_logging.py
src/experiments/experiments_config.py
src/experiments/experiments_train.py
```

CLI files should use the `cli_` prefix:

```text
src/experiments/cli_train.py
src/experiments/cli_evaluate.py
src/experiments/cli_render.py
src/experiments/cli_curriculum.py
```

Do not create generic names such as `helpers.py`, `utils.py`, `main.py`, `misc.py`, or `common.py` unless there is a very clear reason.

Prefer fewer coherent modules over many tiny files.

## Package `__init__.py` Style

Package `__init__.py` files must follow this style.

Use a package-level docstring with this structure:

```python
"""
Short package description.

Provides:
- alias_or_module: concise responsibility description
- another_module: concise responsibility description
"""
```

Then expose stable package aliases with explicit imports and `__all__` when the package has public submodules.

Example pattern:

```python
"""
Trajectory generation and reference path utilities.

Provides:
- primitives: basic geometric trajectory generators
- formations: optional multi-drone formation reference paths
"""

from . import trajectories_primitives as primitives
from . import trajectories_formations as formations

__all__ = [
    "formations",
    "primitives",
]
```

Do not add imports to CLI packages only to make the package look populated.

CLI package `__init__.py` files must stay import-free and use:

```python
"""
Experiment command-line entry points.

Executable modules:
- cli_train: Train a policy from an experiment configuration
- cli_evaluate: Evaluate a saved policy or run directory
- cli_render: Render trajectory tracking examples
"""

__all__: list[str] = []
```

## Module Top-Level Docstring Style

Normal Python modules must use exactly this top-level docstring style structure:

```python
"""
===============================================================================
module_name.py
===============================================================================
One-sentence module purpose.

Responsibilities:
  - Responsibility one
  - Responsibility two
  - Responsibility three

Design principles:
  - Principle one
  - Principle two

Boundaries:
  - What belongs elsewhere
  - What this module must not own

Notes:
  Optional implementation notes when useful with subtitles in the same style as before
===============================================================================

"""
```

If a section is not useful, omit the section entirely rather than adding filler text.

Do not use short one-line module docstrings for non-trivial modules.

Small `__init__.py` files should use the package style above, not the full banner style.

## Function and Class Docstring Style

All public functions, classes, and public methods must have docstrings.

Private helpers should also have docstrings when their behavior is non-obvious, when they encode assumptions, or when they are used by more than one caller.

Simple functions may use a concise one-line docstring:

```python
def get_storage_root() -> Path:
    """Get the storage root directory from environment or default."""
```

Non-trivial functions must use the NumPy-style section format:

```python
def make_tracking_env(config: dict[str, Any], seed: int | None = None) -> gym.Env:
    """
    Build a drone trajectory-tracking environment from a resolved config.

    Parameters
    ----------
    config
        Resolved experiment configuration containing environment, reward and trajectory settings.
    seed
        Optional random seed used for deterministic environment initialization.

    Returns
    -------
    gym.Env
        Configured Gymnasium-compatible drone tracking environment.

    Raises
    ------
    ValueError
        If the configuration contains an unsupported environment or trajectory type.
    """
```

Use these sections when relevant and omit sections that do not apply:

```text
Parameters
----------
Returns
-------
Yields
------
Raises
------
Notes
-----
```

Class docstrings should describe the purpose and the main responsibilities of the class:

```python
class CurriculumManager:
    """
    Coordinate LLM-guided curriculum proposals and validation feedback.

    Parameters
    ----------
    validator
        Task validator used to reject infeasible generated tasks.
    history_limit
        Maximum number of recent curriculum events included in LLM prompts.

    Notes
    -----
    This class does not train the RL policy directly. It only proposes and records curriculum tasks.
    """
```

Do not write placeholder docstrings such as `TODO`, `Docstring`, or `Initialize class`.

Docstrings should document the contract, assumptions, and boundaries, not restate each line of code.

## Source Layout

Current intended source layout:

```text
src/
├── envs/           # Drone environment wrappers, observations, rewards
├── trajectories/   # Trajectory and formation generators
├── llm/            # LLM prompts, schemas, curriculum logic
├── validation/     # Feasibility checks for generated tasks
├── evaluation/     # Metrics, plots, result aggregation
├── experiments/    # Experiment orchestration and CLI entry points
└── utils/          # Paths, seeds, logging, serialization
```

Avoid unnecessary nesting.

Create a subpackage only when several related modules share a stable responsibility.

Prefer fewer coherent modules over many tiny files.

## Scope Rules

The LLM is a curriculum generator, not a low-level drone controller.

The LLM must not generate executable Python code.

All LLM-generated tasks must pass deterministic validation before being used for training or evaluation.

Avoid implementing full multi-agent reinforcement learning unless explicitly requested.

Multi-drone behavior is optional and should be treated as a visualization or showcase extension.

## Notebook Rules

The final notebook is the main report and demonstration artifact.

The notebook should import reusable code from `src/` instead of implementing core logic in notebook cells.

The notebook should not require full retraining by default.

Use flags such as:

```python
TRAIN_FROM_SCRATCH = False
RUN_QUICK_DEMO = True
USE_SAVED_RESULTS = True
```

When asked to provide a notebook check, do not edit the notebook directly unless explicitly requested.

Instead, provide a short manual notebook cell that the user can paste into `Drone_RL_LLM_Curriculum.ipynb`.

A notebook check cell should:

- Import through package aliases, for example `from src import trajectories, validation`
- Run quickly
- Not start training
- Not write large artifacts
- Print or display a small result that verifies the feature manually
- Avoid Pylance errors where practical, for example by narrowing `env.action_space` before using `shape`

## Storage Rules

Large and generated files must not be committed.

Use the external storage directory mounted at:

```text
/workspace/storage
```

A local symlink may expose it as:

```text
repo/storage -> /workspace/storage
```

Generated files belong in:

```text
storage/results/
storage/models/
storage/videos/
storage/gifs/
storage/llm_logs/
storage/wandb/
storage/datasets/
storage/tmp/
```

Only small curated documentation assets should be copied into `docs/`.

## Quality Rules

Use Python 3.10.

Use `uv` for dependency management.

Use Ruff, Mypy, Pytest, and nbQA according to `pyproject.toml`.

Before returning changes, prefer to run:

```bash
ruff check .
ruff format --check .
mypy src
```

If safe automatic fixes are available, apply them before handing work back.

Never commit changes.
