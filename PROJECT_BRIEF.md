# Project Brief: Drone RL with LLM-Guided Curriculum Learning

## Project Goal

This project investigates whether a large language model can act as an adaptive curriculum generator for reinforcement-learning-based quadrotor trajectory tracking.

The core research question is:

> Can an LLM propose valid and useful training tasks that improve the learning process of a drone RL agent compared to a fixed manual curriculum or direct training on hard tasks?

The project is implemented as a modular Python research project with a final Jupyter notebook as the main report and demonstration artifact.

---

## Scope

The project focuses on a single-drone trajectory tracking task in simulation, with optional multi-drone visualization as a showcase.

The main components are:

1. A drone simulation environment based on `gym-pybullet-drones`.
2. A trajectory generation module for hover, line, circle, figure-eight, star, spiral, and related paths.
3. A validation module that checks whether generated tasks are physically and geometrically feasible.
4. An LLM curriculum module that proposes new training tasks in a strict JSON format.
5. A reinforcement-learning training loop using Stable-Baselines3.
6. An evaluation pipeline comparing learning performance, tracking error, success rate, crash rate, and curriculum progression.
7. A final Jupyter notebook explaining the full workflow and showing the results.

---

## Non-Goals

The project does not aim to build a full multi-agent drone swarm RL system.

The LLM must not directly control the drone at runtime.

The LLM must not generate arbitrary executable Python code.

The LLM is only used as a curriculum generator or task proposer. All generated tasks must pass deterministic validation before being used.

Multi-drone behavior is optional and should primarily be used as a visual showcase by applying a trained policy to multiple drones or by rendering predefined formations.

---

## Core Architecture

The repository follows this structure:

```text
src/
├── envs/           # Drone environment wrappers, observations, rewards
├── trajectories/   # Trajectory and formation generators
├── llm/            # LLM prompts, curriculum schemas, repair prompts
├── validation/     # Feasibility checks for generated tasks
├── evaluation/     # Metrics, plots, result aggregation
├── experiments/    # Experiment orchestration and CLI entry points
└── utils/          # Paths, seeds, logging, serialization
```

Shell scripts in `scripts/` are infrastructure only. They are used for Docker image builds, dev-container startup, and GPU job execution. They should not contain core project logic.

The final notebook should import reusable code from `src/` instead of implementing the whole project inside notebook cells.

---

## Expected Experimental Comparison

At minimum, the project should compare:

1. No curriculum / direct training on difficult tasks.
2. Manual or fixed curriculum.
3. LLM-guided adaptive curriculum.

The evaluation should include:

- Learning curves.
- Mean tracking error.
- Success rate.
- Crash rate.
- Energy or action-cost proxy.
- Curriculum difficulty progression.
- Invalid LLM proposal rate.
- Optional repair success rate for invalid LLM tasks.

---

## Curriculum Task Format

LLM-generated tasks should be represented as strict JSON-like structured data.

Example:

```json
{
  "task_type": "trajectory",
  "shape": "figure_eight",
  "num_waypoints": 24,
  "radius": 1.2,
  "height": 1.0,
  "vertical_amplitude": 0.2,
  "speed": 0.7,
  "wind_strength": 0.05,
  "tracking_tolerance": 0.25,
  "reason": "The drone tracks circles reliably, so the next task adds changing curvature while keeping wind low."
}
```

Generated tasks must be validated before use.

Validation should check at least:

- Arena bounds.
- Maximum speed.
- Maximum acceleration.
- Minimum duration.
- No discontinuous jumps.
- Optional multi-drone minimum separation.
- Valid shape and parameter ranges.

---

## LLM Feedback Loop

The LLM should not receive the full training history every time.

Use a compact prompt containing:

- A short global summary.
- Recent accepted tasks.
- Recent invalid tasks and rejection reasons.
- Current evaluation metrics.
- A bounded output schema.

The curriculum system should log all proposals as JSONL:

```text
storage/llm_logs/
```

Each entry should include:

- Proposed task.
- Validation status.
- Rejection reason if invalid.
- Repair attempts if used.
- Training or evaluation metrics if accepted.

---

## Final Notebook

The final notebook is the main submission artifact.

It should include:

1. Introduction and motivation.
2. Environment description.
3. State, action, and reward definition.
4. RL algorithm description.
5. LLM curriculum design.
6. Validation and safety checks.
7. Training setup.
8. Evaluation protocol.
9. Learning curves and quantitative results.
10. Visual trajectory/showcase examples.
11. Discussion.
12. Conclusion and outlook.

The notebook should not require full retraining by default. It should be able to load saved results and models from `storage/`.

Recommended notebook flags:

```python
TRAIN_FROM_SCRATCH = False
RUN_QUICK_DEMO = True
USE_SAVED_RESULTS = True
```

---

## Storage Policy

Large or generated files must not be committed to Git.

The external storage directory is mounted inside the container at:

```text
/workspace/storage
```

A local symlink may expose it as:

```text
repo/storage -> /workspace/storage
```

The following outputs belong in storage:

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

Only small curated figures or media that are needed for the README or final report should be copied into `docs/`.

---

## Development Rules

Use Python 3.10.

Use `uv` for dependency management.

Use Ruff, Mypy, Pytest, and nbQA according to `pyproject.toml`.

Keep the code modular and importable.

Avoid putting large logic blocks directly into the notebook.

Prefer deterministic seeds where possible.

Keep all generated artifacts out of Git unless they are small documentation assets.
