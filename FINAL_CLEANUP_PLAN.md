# Final Cleanup Plan

This document is an audit and implementation plan only. It does not implement the refactor, move files, rename modules, edit notebooks, or create commits.

## 1. Executive summary

The repository already has a useful foundation for a drone reinforcement-learning project: validated trajectory tasks, a Gymnasium/PyBullet tracking environment, PPO training, policy diagnostics, policy/scenario rendering, manual curriculum training, and curriculum evaluation. The core issue is that the orchestration layer has grown from smoke/MVP helpers into production-facing experiment code while remaining flat under `src/experiments`.

The main cleanup goal is to stabilize the final research workflow around four comparable run types:

- direct PPO baseline training
- fixed/manual curriculum training
- LLM-guided adaptive curriculum training
- evaluation and comparison across shared benchmark suites

The recommended direction is to keep the top-level package name `src/experiments` but split it into clear subpackages for `cli`, `training`, `evaluation`, `curriculum`, `rendering`, and `comparison`. This preserves the current project import style while making room for report-ready artifacts, evaluation suites, a final notebook contract, and future LLM curriculum integration.

Working tree note: the repository currently has uncommitted edits in `src/evaluation/evaluation_diagnostics.py`, `src/validation/validation_tasks.py`, `tests/test_envs_task_adapter.py`, `tests/test_envs_tracking_env.py`, and `tests/test_evaluation_diagnostics.py`. The cleanup/refactor should not start until those runtime bugfixes are validated or deliberately parked.

## 2. Current architecture map

### `PROJECT_BRIEF.md`

`PROJECT_BRIEF.md` describes the intended project: single-drone trajectory tracking, deterministic task validation, PPO training, manual and LLM-guided curricula, evaluation metrics, and a final notebook that loads saved results from storage by default. It also establishes that the LLM is a task proposer only, not a controller and not a source of executable Python code.

### `README.md`

`README.md` is currently a high-level overview. It describes the research question, Docker workflow, intended source layout, and final notebook. It does not yet document the final run contract, final evaluation suite, final comparison outputs, or result GIFs/plots.

### `AGENTS.md`

`AGENTS.md` defines strict local workflow rules: read `PROJECT_BRIEF.md` before changes, use scoped escalation for shell commands in this container, avoid commits, keep core logic in `src/`, keep scripts as infrastructure, preserve package-prefix file naming, and do not edit notebooks unless explicitly requested. The plan below follows those constraints.

### `pyproject.toml`

`pyproject.toml` defines Python 3.10 to <3.12, uses `setuptools`, and includes `numpy`, `scipy`, `pandas`, `matplotlib`, `torch`, `gymnasium`, `stable-baselines3[extra]`, `gym-pybullet-drones`, `wandb`, `optuna`, `requests`, `pydantic`, `jsonschema`, `imageio`, `opencv-python-headless`, and dev tools. Ruff is broad but has research-friendly ignores. Mypy is configured for `src`. Pytest looks under `tests`.

### `configs/`

Current config files are:

- `configs/smoke/trajectory_validation.yaml`: validation smoke task catalog with hover, circle, line, vertical, and polyline tasks plus validation limits.
- `configs/smoke/training_smoke.yaml`: deterministic MVP training-smoke settings.
- `configs/training/ppo_tracking.yaml`: PPO tracking config with task selection, total timesteps, eval steps, seed, action normalization, and W&B fields. It does not yet contain a nested `ppo:` hyperparameter block.
- `configs/curricula/manual_line_curriculum.yaml`: five-stage manual line curriculum with explicit tasks, per-stage timesteps, and model transfer through the current PPO helper.
- `configs/evaluation/curriculum_benchmarks.yaml`: named curriculum benchmark tasks for line, polyline, circle, and long line.
- `configs/scenarios/*.yaml`: scenario render configurations for scripted reference and PPO showcase rollouts.

Missing final configs include direct-vs-curriculum comparison config, final benchmark suite config, and a general evaluation suite schema.

### `scripts/`

`scripts/docker_build.sh`, `scripts/docker_dev.sh`, `scripts/docker_job.sh`, and `scripts/_docker_run.sh` are Docker/HPC wrappers. They create storage directories, mount `/workspace/storage`, set environment variables, and run Python modules or scripts. They should remain infrastructure only. They currently know about `training_runs`, `evaluation_runs`, and `comparison_reports`.

### `src/envs/`

`src/envs` is compact and should remain a stable core package.

- `envs_builders.py`: minimal upstream HoverAviary constructor for smoke and tracking wrappers.
- `envs_task_adapter.py`: converts validated trajectory task dictionaries into immutable environment reference data.
- `envs_tracking_env.py`: Gymnasium-compatible `TrajectoryTrackingEnv` and normalized action wrapper for PPO.
- `envs_tracking_reward.py`: deterministic tracking reward and per-step diagnostic helpers.

The main cleanup need is terminology: several docstrings still call behavior MVP/smoke even though these modules are now core project infrastructure.

### `src/trajectories/`

`src/trajectories` currently has only `trajectories_primitives.py`, which supports hover, circle, line, vertical, and polyline trajectories. The brief mentions figure-eight, star, spiral, and optional formations, but those are not present yet. Do not add them as part of the structural refactor unless a final evaluation suite needs them.

### `src/validation/`

`src/validation` contains shared task vocabulary and deterministic feasibility checks.

- `validation_contracts.py`: supported task types, shapes, and field constants. It includes manual curriculum shapes such as `hover_stabilization`, `nearby_target_hover`, `start_hold_then_short_line`, and `short_slow_line`.
- `validation_tasks.py`: validates task dictionaries and sampled trajectories, including start-hold metadata and motion/bounds limits.

This package should stay separate from experiments. LLM and curriculum code should consume it, not duplicate validation logic.

### `src/evaluation/`

`src/evaluation` contains reusable evaluation logic.

- `evaluation_trajectory_metrics.py`: trajectory tracking error metrics.
- `evaluation_rollout.py`: deterministic rollout evaluation and JSONL trace helpers.
- `evaluation_plots.py`: trajectory and rollout trace plots.
- `evaluation_diagnostics.py`: trained-policy diagnostics, failure-mode classification, and curriculum feedback JSON artifacts.

This package is useful and should remain domain logic, while run orchestration and suite selection should live under `src/experiments/evaluation`.

### `src/experiments/`

`src/experiments` is currently too large and flat: 19 files and about 8.2k lines. It mixes CLIs, smoke helpers, PPO training, policy evaluation, policy rendering, scenario rendering, manual curriculum training, and curriculum evaluation. `experiments_ppo_tracking.py`, `experiments_policy_render.py`, and `experiments_scenario_render.py` are each large enough to justify either subpackages or smaller internal splits.

The package name `experiments` is still acceptable because this is a research project and the package owns experiment orchestration. The problem is not the name alone. The problem is the flat structure and lingering smoke terminology in paths that now run the main baseline and curriculum workflows.

### `src/utils/`

`src/utils` contains:

- `utils_paths.py`: legacy/category storage roots such as `results`, `models`, `gifs`, `llm_logs`, and `wandb`.
- `utils_artifacts.py`: newer run-scoped helpers for `training_runs`, `evaluation_runs`, and `comparison_reports`.
- `utils_wandb.py`: optional W&B initialization, grouped summary metrics, and artifact logging.

The main issue is that `utils_paths.py` and `utils_artifacts.py` express different artifact layouts. The final plan should consolidate on one run container and keep compatibility wrappers only during migration.

### `src/llm/`

`src/llm` currently contains only `llm_task_schema.py`. It can build a compact schema-like contract, normalize proposed task mappings, and validate proposed tasks through deterministic validation. It does not yet include LLM prompts, parsing of model responses, repair, event logging, LLM client boundaries, or a curriculum runner.

Also note that `llm_task_schema.py` does not yet expose every curriculum-specific validation field from `validation_contracts.py` as a first-class prompt/schema option. That is acceptable for the current MVP but should be revisited before LLM curriculum work.

### `tests/`

Tests are broad for the current MVP: environment adapters, tracking env, reward, trajectories, validation, diagnostics, plots, rollout, artifact paths, W&B, PPO tracking, render helpers, scenario rendering, manual curriculum training/evaluation, CLI help, and LLM task schema. Missing test areas align with missing final features: explicit PPO hyperparameter config, evaluation suite loading, unified run layout, comparison outputs, LLM parser/repair/event logs, and notebook smoke/nbQA after the notebook is real.

### Notebook

`Drone_RL_LLM_Curriculum.ipynb` currently contains one markdown title cell. It is not yet the final report artifact and does not yet define `TRAIN_FROM_SCRATCH`, `RUN_QUICK_DEMO`, or `USE_SAVED_RESULTS`.

## 3. `src/experiments` file inventory

| file | current responsibility | CLI or library module | keep / rename / move / merge / delete candidate | proposed target location | proposed target filename | reason |
|---|---|---:|---|---|---|---|
| `src/experiments/__init__.py` | Exposes aliases for config, curriculum, curriculum training/evaluation, policy evaluation/rendering, PPO tracking, render smoke, scenario render, and training smoke. | Library package initializer | Keep and update | `src/experiments/` | `__init__.py` | Keep package-level aliases, but re-export from subpackages after restructuring. Do not import CLI modules here. |
| `src/experiments/cli_evaluate_curriculum.py` | CLI for curriculum own-stage, benchmark, and generalization evaluation. | CLI | Rename and move | `src/experiments/cli/` | `experiments_cli_evaluate_curriculum.py` | Thin CLI should live with other entry points. Keep separate from generic policy evaluation because it parses curriculum summary and benchmark options. |
| `src/experiments/cli_mvp.py` | Runs or prints the old MVP smoke sequence. | CLI | Delete candidate after final comparison CLI exists; temporary move if still needed | `src/experiments/cli/` temporarily | `experiments_cli_mvp.py` | Useful as a reviewer smoke path today, but it overlaps with final direct PPO/evaluation/render/comparison commands and should not be a final research workflow. |
| `src/experiments/cli_render_policy.py` | CLI for rendering a trained PPO policy on one task. | CLI | Rename and move | `src/experiments/cli/` | `experiments_cli_render_policy.py` | Keep as a report/showcase entry point, but remove smoke language from help/docstrings. |
| `src/experiments/cli_render_scenario.py` | CLI for continuous multi-phase scenario rendering with PPO or scripted-reference controller. | CLI | Rename and move | `src/experiments/cli/` | `experiments_cli_render_scenario.py` | Keep because scenario rendering is a distinct showcase workflow. It should call public rendering APIs rather than private helpers. |
| `src/experiments/cli_render_smoke.py` | CLI for tiny headless render smoke rollout. | CLI | Delete candidate or move as debug-only | `src/experiments/cli/` if retained | `experiments_cli_render_smoke.py` | This is a true smoke/integration helper, not a final report workflow. Retain only if CI or Docker verification needs it. |
| `src/experiments/cli_train_curriculum.py` | CLI for manual curriculum training. | CLI | Rename and move | `src/experiments/cli/` | `experiments_cli_train_curriculum.py` | Keep, but eventually support `--kind manual` or separate manual/LLM CLIs once LLM curriculum exists. |
| `src/experiments/cli_train_tracking.py` | CLI for PPO trajectory tracking. Currently described as smoke training but used as direct PPO baseline. | CLI | Rename and move | `src/experiments/cli/` | `experiments_cli_train_tracking.py` | This is part of the final direct PPO baseline. Rename docs/help away from smoke terminology. |
| `src/experiments/cli_training_smoke.py` | CLI for deterministic synthetic training smoke loop. | CLI | Delete candidate after Phase 1/2; temporary move only if tests still depend on it | `src/experiments/cli/` temporarily | `experiments_cli_training_smoke.py` | Overlaps with real PPO baseline and should not be a final training path. Keep only as a fast CI smoke command if still useful. |
| `src/experiments/experiments_config.py` | Loads YAML config files into plain dictionaries. | Library | Keep | `src/experiments/` | `experiments_config.py` | Small shared loader is fine at package root. Later schema-specific loaders should live in training/evaluation/curriculum modules. |
| `src/experiments/experiments_curriculum.py` | Summarizes configured task lists and validation results. | Library | Rename and move; possible merge later | `src/experiments/curriculum/` | `experiments_curriculum_validation.py` | It is not a curriculum runner; it is a config/task validation summary helper. Keep if used by notebook/README checks; otherwise merge into curriculum config validation tests. |
| `src/experiments/experiments_curriculum_evaluation.py` | Evaluates curriculum stages against own-stage or named benchmark tasks through shared policy evaluation. | Library | Move and keep | `src/experiments/curriculum/` | `experiments_curriculum_evaluation.py` | Curriculum-specific aggregation belongs with curriculum. Benchmark suite parsing may later move to `experiments_evaluation_suites.py`. |
| `src/experiments/experiments_curriculum_training.py` | Sequential manual curriculum training with per-stage task configs, model transfer, summaries, and manifests. | Library | Rename and move | `src/experiments/curriculum/` | `experiments_curriculum_manual.py` | This is specifically manual curriculum orchestration. It should be separated from future LLM curriculum while sharing stage runner utilities. |
| `src/experiments/experiments_policy_evaluation.py` | Shared policy evaluation pipeline that writes diagnostics, traces, plots, renders, metrics, and manifests for one model/task. | Library | Rename and move | `src/experiments/evaluation/` | `experiments_evaluation_policy.py` | This is the central reusable evaluation API. Keep separate from curriculum aggregation and comparison. |
| `src/experiments/experiments_policy_render.py` | Trained-policy rollout rendering, camera capture, GIFs, traces, plots, manifests, and visual overlays. | Library | Rename, move, and simplify | `src/experiments/rendering/` | `experiments_rendering_policy.py` | Keep separate from evaluation metrics, but expose public rollout/render helpers so evaluation and scenario rendering stop calling private functions. |
| `src/experiments/experiments_ppo_tracking.py` | Real PPO tracking training, model saving, metrics, W&B, liftoff diagnostics, and post-train evaluation. Still named and modeled as smoke. | Library | Rename, move, and split | `src/experiments/training/` | `experiments_training_ppo_tracking.py` plus `experiments_training_ppo_config.py` | This is now the direct PPO baseline and curriculum stage trainer. Remove smoke naming and extract PPO hyperparameter/config handling. Liftoff diagnostics may deserve `experiments_training_liftoff_diagnostics.py`. |
| `src/experiments/experiments_render_smoke.py` | Tiny render smoke/integration helper with fallback plot. | Library | Keep as debug-only or delete candidate | `src/experiments/rendering/` if retained | `experiments_rendering_smoke.py` | This is a legitimate smoke helper, but it should not be part of the final report path unless CI uses it. |
| `src/experiments/experiments_scenario_render.py` | Continuous multi-phase scenario composition and rendering with PPO or scripted-reference controller. | Library | Rename, move, and split composition from rendering | `src/experiments/rendering/` | `experiments_rendering_scenario.py` and possibly `experiments_rendering_scenario_composition.py` | It is the largest module and owns both scenario geometry composition and simulator rendering. Split only where it reduces real complexity. |
| `src/experiments/experiments_training_smoke.py` | Synthetic deterministic training-smoke loop without real PPO. | Library | Delete candidate; temporary move only for CI | `src/experiments/training/` temporarily | `experiments_training_smoke.py` | Historical MVP helper. It overlaps conceptually with real training and should not remain a final training module. |

Files that should remain separate:

- `experiments_training_ppo_tracking.py` and `experiments_evaluation_policy.py`: training and evaluation should not be merged.
- `experiments_curriculum_manual.py` and future `experiments_curriculum_llm.py`: fixed curricula and LLM-guided curricula should share stage utilities but keep proposal logic separate.
- `experiments_rendering_policy.py` and `experiments_rendering_scenario.py`: one-task policy rendering and multi-phase scenario rendering have different contracts.

Files that can be merged or retired:

- `experiments_training_smoke.py` and `cli_training_smoke.py` can be retired after real PPO config and fast tests are reliable.
- `cli_mvp.py` can be retired after `experiments_cli_compare_runs.py` and final notebook loading are stable.
- `experiments_curriculum.py` can be merged into curriculum config validation if the notebook no longer needs a standalone summary helper.
- Benchmark loading in `experiments_curriculum_evaluation.py` can move to shared evaluation suite loading once non-curriculum policy evaluation uses the same suite schema.

## 4. Problems found

1. `src/experiments` is flat and overloaded. It contains CLIs, training, evaluation, curriculum, rendering, and smoke utilities in one package directory.

2. CLI code and library orchestration are mixed. Files named `cli_*.py` sit beside large reusable modules, making it hard to scan public APIs and increasing import churn.

3. The top-level name `experiments` is broad but still acceptable. Renaming it to `workflows`, `pipelines`, `runs`, or `orchestration` would cause heavy import churn and conflict with the current brief. Subpackages solve the real problem with less risk.

4. Smoke naming is now inaccurate in production paths. `experiments_ppo_tracking.py`, `PPOTrackingSmokeSettings`, `PPOTrackingSmokeResult`, `run_ppo_tracking_smoke`, and `mode: ppo_smoke` are used by direct PPO and manual curriculum training. Those should become production training names. True smoke helpers should either remain explicitly smoke-only or be deleted.

5. PPO hyperparameters are hardcoded in `experiments_ppo_tracking.py`. Current code directly passes `policy="MlpPolicy"`, `device="cpu"`, `gamma=0.95`, `learning_rate=1.0e-3`, `n_epochs=4`, `n_steps=rollout_steps`, and `batch_size=rollout_steps` into `PPO(...)`.

6. Hidden PPO rollout limits exist. `_MIN_PPO_ROLLOUT_STEPS = 2`, `_MAX_PPO_ROLLOUT_STEPS = 64`, and `_ppo_rollout_steps(total_timesteps)` silently cap `n_steps` and batch size. This makes final training behavior hard to reproduce from config.

7. Training and evaluation are too coupled. `run_ppo_tracking_smoke()` trains, saves, runs liftoff diagnostics, performs post-train policy evaluation, writes metrics, writes manifests, and logs W&B. Final evaluation suites should be runnable independently against saved models.

8. Curriculum-specific duplication is emerging. Manual curriculum training writes stage configs and summaries, while curriculum evaluation writes benchmark task configs and aggregate summaries with its own path logic. A shared stage/run manifest contract would reduce duplication.

9. Path and artifact helpers are inconsistent. `utils_paths.py` exposes category roots like `storage/results` and `storage/models`, while `utils_artifacts.py` exposes `storage/training_runs`, `storage/evaluation_runs`, and `storage/comparison_reports`. Some modules also special-case paths containing `results`.

10. W&B grouping and config are not final. Direct PPO uses groups like `ppo_tracking/<shape>`, manual curriculum uses `curriculum/<curriculum_name>`, and curriculum evaluation accepts W&B mode only for CLI symmetry. PPO hyperparameters are not stored as a complete config object in W&B.

11. README and notebook contract are missing. The README is still descriptive, and the notebook is a one-cell stub. Neither explains the final artifact paths the report should load.

12. Tests are strong for existing MVP modules but weak for final contracts. Missing areas include explicit PPO config validation, evaluation suite configs, unified run layout, comparison reports, LLM parser/repair logs, and notebook smoke/nbQA.

13. The LLM curriculum gap is substantial. Only deterministic task schema helpers exist. There is no prompt builder, response parser, repair pipeline, proposal event log, LLM runner, or integration with curriculum stages.

14. The comparison pipeline is missing. There is no module or CLI to aggregate direct PPO, manual curriculum, and LLM curriculum runs into one JSON/CSV/plot bundle for the final notebook.

15. Rendering modules call private helpers across modules. `experiments_policy_evaluation.py` and `experiments_scenario_render.py` call private functions from `experiments_policy_render.py`. Those should become public helpers or move into shared rendering utilities.

16. Planned trajectory variety is incomplete. Current primitives cover hover, circle, line, vertical, and polyline. Figure-eight, star, spiral, and formation/showcase utilities are not implemented yet. Add only if final experiments need them.

## 5. Recommended final package structure

### Option A: keep `src/experiments` flat, but cleaned up

Example:

```text
src/experiments/
  experiments_config.py
  experiments_training_ppo_tracking.py
  experiments_evaluation_policy.py
  experiments_curriculum_manual.py
  experiments_curriculum_llm.py
  experiments_rendering_policy.py
  experiments_comparison_runs.py
  experiments_cli_train_tracking.py
```

Pros:

- Minimal import-path depth.
- Lowest move churn.

Cons:

- The folder is already too large and would keep growing.
- CLIs, training, evaluation, rendering, and curriculum would remain visually mixed.
- Large modules would still be difficult to navigate.

Verdict: not recommended.

### Option B: keep `src/experiments`, add subpackages

Recommended structure:

```text
src/experiments/
  __init__.py
  experiments_config.py
  cli/
    __init__.py
    experiments_cli_train_tracking.py
    experiments_cli_train_curriculum.py
    experiments_cli_evaluate_policy.py
    experiments_cli_evaluate_curriculum.py
    experiments_cli_render_policy.py
    experiments_cli_render_scenario.py
    experiments_cli_compare_runs.py
  training/
    __init__.py
    experiments_training_ppo_config.py
    experiments_training_ppo_tracking.py
    experiments_training_liftoff_diagnostics.py
  evaluation/
    __init__.py
    experiments_evaluation_policy.py
    experiments_evaluation_suites.py
  curriculum/
    __init__.py
    experiments_curriculum_validation.py
    experiments_curriculum_manual.py
    experiments_curriculum_llm.py
    experiments_curriculum_events.py
  rendering/
    __init__.py
    experiments_rendering_policy.py
    experiments_rendering_scenario.py
    experiments_rendering_scenario_composition.py
  comparison/
    __init__.py
    experiments_comparison_runs.py
    experiments_comparison_plots.py
```

Pros:

- Keeps the project import style and `PROJECT_BRIEF.md` architecture intact.
- Separates CLI entry points from importable orchestration.
- Gives large modules natural homes without a global top-level rename.
- Supports final direct, manual, LLM, evaluation, rendering, and comparison workflows.

Cons:

- Requires import updates and compatibility planning.
- Tests that import old package aliases need coordinated changes.

Verdict: recommended.

### Option C: add top-level `src/cli`, keep nested experiment subpackages

Example:

```text
src/cli/
  cli_train_tracking.py
  cli_evaluate_policy.py
  cli_compare_runs.py
src/experiments/
  training/
  evaluation/
  curriculum/
  rendering/
  comparison/
```

Pros:

- CLIs are easy to find.
- CLI file names can stay short with the `cli_` prefix.

Cons:

- Splits experiment entry points away from their orchestration package.
- Adds a new top-level public namespace.
- Requires Docker/HPC command updates from `src.experiments.cli_*` to `src.cli.*`.
- Less consistent with the current brief and package alias style.

Verdict: acceptable but not recommended for this repository.

### Option D: rename/reframe `experiments` to `workflows`, `pipelines`, `runs`, or `orchestration`

Pros:

- `workflows` or `pipelines` might describe orchestration more precisely.
- `runs` might align with artifact layout.

Cons:

- Large import churn with little functional gain.
- Conflicts with `PROJECT_BRIEF.md`, `AGENTS.md`, README, tests, and current package aliases.
- Still requires subpackages to solve the flat-folder problem.

Verdict: reject for final cleanup. Keep `experiments` and make it structured.

### Concrete structure decision

Use Option B. Keep `src/experiments` as the orchestration namespace and add subpackages. The first restructuring pass should be behavior-preserving: move/rename files, update imports, and keep compatibility aliases where helpful. Do not combine this with PPO config extraction or artifact layout migration.

## 6. Final run/artifact structure

### Current layout

Current helpers and scripts write under:

```text
storage/training_runs/<run_name>/
storage/evaluation_runs/<run_name>/
storage/comparison_reports/<run_name>/
```

Older helpers also expose:

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

The split makes it hard for the final notebook to answer a simple question: "given one run, where are its config, model, training metrics, evaluations, renders, and manifests?"

### Recommended normal run layout

Move new outputs to one run container:

```text
storage/runs/<run_name>/
  run_manifest.json
  config/
    training_config.yaml
    task_config.yaml
    evaluation_suites/
  training/
    models/
    metrics/
    diagnostics/
    logs/
    wandb/
    manifest.json
  evaluations/
    <evaluation_name>/
      diagnostics/
      traces/
      plots/
      renders/
      metrics/
      manifests/
```

Compatibility plan:

- Add new helpers in `utils_artifacts.py` for `storage/runs/<run_name>`.
- Keep existing `get_training_run_dir()` and `get_evaluation_run_dir()` as transitional wrappers only until all current tests and scripts are migrated.
- Do not bulk-move old generated artifacts unless the user explicitly requests storage migration.
- Update the final notebook to load only from the new `storage/runs` contract.

### Curriculum layout options

Option 1:

```text
storage/runs/<curriculum_run_name>/
  run_manifest.json
  config/
  stages/
    stage01_<stage_name>/
      training/
      evaluations/
    stage02_<stage_name>/
      training/
      evaluations/
```

Option 2:

```text
storage/runs/<curriculum_run_name>/
  stages/
    stage01_<stage_name>/
      training/
      evaluations/
  evaluations/
```

Option 3:

```text
storage/runs/<curriculum_run_name>/
  stages/
  curriculum_summary/
```

Recommended: Option 1.

Reasoning:

- Each stage owns its trained model and the evaluations that generated feedback for the next stage.
- The final model is the model produced by the last stage.
- The final model evaluation should usually be the evaluation stored under the last stage.
- If a separate final benchmark suite is needed, store it as `stages/stageNN_<name>/evaluations/final_benchmark/` rather than a top-level `final_evaluations/` folder.
- A top-level `curriculum_summary/` folder is not needed by default. Use `run_manifest.json` for cross-stage pointers and store per-stage metrics in stage manifests.

When a summary is justified:

- LLM curriculum can produce unique cross-stage data that is not duplicated by stage manifests: proposal count, invalid proposal rate, repair attempts, accepted/rejected task history, prompt model metadata, and compact rationale history.
- Store that as root-level manifest fields plus `curriculum_events.jsonl` or `llm/events.jsonl`, not as a separate `curriculum_summary/` directory unless the files become numerous.

## 7. PPO configuration plan

### Proposed config schema

Update `configs/training/ppo_tracking.yaml` toward:

```yaml
run_name: null
seed: 0
check_env: true
normalize_actions: true

task:
  task_config_path: configs/smoke/trajectory_validation.yaml
  task_shape: hover
  task_index: 0

training:
  total_timesteps: 4096
  eval_after_training: false

ppo:
  policy: MlpPolicy
  device: cpu
  learning_rate: 0.0003
  gamma: 0.99
  gae_lambda: 0.95
  n_steps: 256
  batch_size: 64
  n_epochs: 5
  clip_range: 0.2
  ent_coef: 0.001
  vf_coef: 0.5
  max_grad_norm: 0.5
  target_kl: 0.03

wandb:
  mode: auto
  project: drone-rl-llm-curriculum
  entity: null
  group: null
  name: null
  tags: []
```

A transition can support the old flat keys temporarily, but all new configs should use nested sections.

### Where to define the schema

Create:

```text
src/experiments/training/experiments_training_ppo_config.py
```

Suggested public API:

- `PPOConfig`: dataclass or Pydantic model for SB3 PPO hyperparameters.
- `TrackingTrainingConfig`: task, seed, run, W&B, and training budget settings.
- `load_tracking_training_config(path: str | Path) -> TrackingTrainingConfig`
- `ppo_kwargs(config: PPOConfig) -> dict[str, Any]`

The repo already uses dataclasses heavily, so a dataclass with explicit validation is the lowest-risk first step. Pydantic can be used later if validation complexity grows.

### How to validate it

Validate at load time:

- `policy` must be non-empty.
- `device` must be non-empty and ideally one of `cpu`, `cuda`, or `auto`.
- `learning_rate`, `gamma`, `gae_lambda`, `clip_range`, `vf_coef`, and `max_grad_norm` must be finite and within meaningful ranges.
- `n_steps`, `batch_size`, and `n_epochs` must be positive integers.
- `batch_size <= n_steps` unless explicitly justified.
- `total_timesteps >= n_steps` for normal runs. Tiny tests should set tiny `ppo.n_steps` explicitly instead of relying on hidden caps.

### How to pass it into `PPO(...)`

Replace the hardcoded constructor with explicit kwargs:

```python
ppo_config = training_config.ppo
ppo_kwargs = ppo_config.to_sb3_kwargs()
policy = ppo_kwargs.pop("policy")
model = PPO(
    policy,
    training_env,
    seed=training_config.seed,
    tensorboard_log=str(logs_dir),
    verbose=0,
    **ppo_kwargs,
)
```

### How to store it

Store the complete resolved PPO config in:

- training metrics JSON under `ppo_config`
- training manifest under `ppo_config`
- W&B config under `ppo`
- copied run config under `storage/runs/<run_name>/config/training_config.yaml`

Also store `ppo_effective_config` if SB3 receives transformed values such as schedules.

### What to do with current `rollout_steps` and hidden `n_steps` limits

Remove `_ppo_rollout_steps()` as a hidden production behavior. Keep a temporary compatibility test if needed, but new training should use `ppo.n_steps` directly.

For fast smoke or CI runs, set explicit small values in the test config:

```yaml
training:
  total_timesteps: 64
ppo:
  n_steps: 16
  batch_size: 16
```

This makes the rollout length visible and reproducible.

## 8. Training/evaluation separation plan

Separate four responsibilities:

1. Training a model.
2. Evaluating a saved model.
3. Rendering a rollout.
4. Aggregating comparison results.

### Desired training API

Target module:

```text
src/experiments/training/experiments_training_ppo_tracking.py
```

Public functions:

- `train_ppo_tracking(config: TrackingTrainingConfig) -> TrainingRunResult`
- `train_ppo_tracking_from_config(path: str | Path, overrides: TrainingOverrides | None = None) -> TrainingRunResult`

Training should:

- load and validate task config
- build the environment
- create or load a PPO model
- run `model.learn(...)`
- save model, training metrics, logs, and manifest
- optionally run a small health diagnostic if configured

Training should not own final benchmark evaluation or report rendering.

### Desired evaluation API

Target modules:

```text
src/experiments/evaluation/experiments_evaluation_policy.py
src/experiments/evaluation/experiments_evaluation_suites.py
```

Public functions:

- `run_policy_evaluation(spec: PolicyEvaluationSpec, artifacts: PolicyEvaluationArtifactOptions | None = None) -> PolicyEvaluationResult`
- `load_evaluation_suite(path: str | Path) -> EvaluationSuite`
- `run_evaluation_suite(run_manifest: str | Path, suite: EvaluationSuite) -> EvaluationSuiteResult`

### Desired rendering API

Target modules:

```text
src/experiments/rendering/experiments_rendering_policy.py
src/experiments/rendering/experiments_rendering_scenario.py
```

Rendering should:

- expose public rollout/camera/GIF helpers used by evaluation
- keep visual overlays and simulator capture out of training modules
- write GIFs and plots under evaluation artifact directories

### Desired comparison API

Target module:

```text
src/experiments/comparison/experiments_comparison_runs.py
```

Public functions:

- `compare_runs(config: ComparisonConfig) -> ComparisonResult`
- `load_comparison_config(path: str | Path) -> ComparisonConfig`

Comparison should not train or evaluate models. It should consume already-written run and evaluation manifests.

## 9. Evaluation suites plan

Create config files such as:

```text
configs/evaluation/line_eval_suite.yaml
configs/evaluation/curriculum_stage_eval_suite.yaml
configs/evaluation/final_benchmark_eval_suite.yaml
```

Suggested schema:

```yaml
evaluation_name: final_benchmark
seed: 0
eval_steps: 240
render:
  enabled: true
  fps: 20
  max_steps: null
plots:
  enabled: true
traces:
  enabled: true
expected_outputs:
  metrics_filename: evaluation_summary.json
  manifest_filename: evaluation_manifest.json
tasks:
  - task_name: line_basic
    task_shape: line
    task:
      task_type: trajectory
      shape: line
      duration_sec: 3.0
      sample_rate_hz: 10.0
      start_hold_enabled: true
      start_hold_sec: 1.0
      exclude_start_hold_from_tracking_metrics: true
      start: [0.0, 0.0, 1.0]
      end: [1.0, 0.0, 1.0]
```

Suite loader responsibilities:

- validate `evaluation_name`
- validate `seed` and `eval_steps`
- validate every task through `validation.tasks.validate_task`
- expand per-task output names deterministically
- return a suite object consumed by policy, curriculum, and comparison workflows

Existing `configs/evaluation/curriculum_benchmarks.yaml` should either be migrated into this schema or loaded through a compatibility adapter.

## 10. Curriculum plan

### Final intended architecture

Manual curriculum:

```text
src/experiments/curriculum/experiments_curriculum_manual.py
```

LLM curriculum:

```text
src/experiments/curriculum/experiments_curriculum_llm.py
```

Shared event/stage helpers:

```text
src/experiments/curriculum/experiments_curriculum_events.py
```

### Manual curriculum runner

The manual runner should:

- load a fixed stage config
- validate all stages before training starts
- train each stage through the shared PPO training API
- transfer the previous stage model into the next stage when configured
- run a configured stage evaluation suite after each stage
- write each stage under `storage/runs/<curriculum_run>/stages/stageNN_<stage_name>/`

### LLM curriculum runner

The LLM runner should:

- start from an initial task or seed stage
- after each stage, read compact evaluation feedback
- build a bounded prompt from recent accepted tasks, recent invalid tasks, and current metrics
- parse exactly one JSON task proposal
- validate through `validation.tasks.validate_task`
- optionally attempt repair through a deterministic/LLM repair step
- log accepted, rejected, and repaired proposals
- train the next stage only after validation succeeds

The LLM must not generate executable code.

### What happens after each stage

After each stage:

1. Save the stage model and training manifest.
2. Run the stage evaluation suite.
3. Write diagnostics and curriculum feedback.
4. Record a stage event containing task, model path, evaluation summary, and next-task decision inputs.
5. For manual curricula, advance to the next configured task.
6. For LLM curricula, request and validate the next task.

### Final model and final evaluation

The last stage model is the final curriculum model. Extra final evaluation is needed only when the final benchmark suite differs from the stage evaluation suite. If needed, store it under the final stage:

```text
storage/runs/<curriculum_run>/stages/stageNN_<stage_name>/evaluations/final_benchmark/
```

Do not add top-level `final_evaluations/` unless a future workflow evaluates multiple final-stage model variants outside the stage structure.

## 11. LLM curriculum gap

Current `src/llm` contains only:

```text
src/llm/llm_task_schema.py
```

It provides schema-like task metadata, prompt-contract text, normalization, and deterministic validation. Missing pieces:

- prompt templates using compact history and evaluation feedback
- strict response parser for JSON-only LLM output
- repair prompt or deterministic repair logic
- proposal event schema
- JSONL logging for accepted/rejected/repaired proposals
- LLM client abstraction and offline/mock mode for tests
- integration with curriculum stage training

Proposed modules:

```text
src/llm/llm_task_schema.py
src/llm/llm_curriculum_prompts.py
src/llm/llm_task_parser.py
src/llm/llm_task_repair.py
src/llm/llm_curriculum_client.py
src/experiments/curriculum/experiments_curriculum_llm.py
```

Also update `src/llm/__init__.py` with package aliases once those modules exist.

## 12. Comparison pipeline plan

The comparison pipeline should answer the final research question directly: direct PPO vs manual curriculum vs LLM-guided curriculum.

### Inputs

A comparison config should point to run manifests, not raw model paths:

```yaml
comparison_name: final_direct_manual_llm_seed0
evaluation_suite: configs/evaluation/final_benchmark_eval_suite.yaml
runs:
  - label: direct_ppo
    kind: direct_ppo
    run_manifest: storage/runs/direct_ppo_line_seed0/run_manifest.json
  - label: manual_curriculum
    kind: manual_curriculum
    run_manifest: storage/runs/manual_line_v1_seed0/run_manifest.json
  - label: llm_curriculum
    kind: llm_curriculum
    run_manifest: storage/runs/llm_curriculum_v1_seed0/run_manifest.json
```

### Outputs

Write under:

```text
storage/runs/<comparison_run_name>/comparison/
  metrics/
    comparison_summary.json
    comparison_rows.csv
  plots/
    tracking_error_bars.png
    success_crash_rates.png
    curriculum_progression.png
  manifests/
    comparison_manifest.json
```

### Metrics to aggregate

- mean, final, and max position error
- success rate or task completion rate
- crash/termination/truncation counts
- action-cost proxy and saturation fraction
- total timesteps and number of curriculum stages
- time or sample budget to reach threshold, if available
- invalid LLM proposal rate
- repair success rate
- final benchmark score per task

### Notebook contract

The notebook should load `comparison_summary.json`, `comparison_rows.csv`, and the comparison plot files. It should not recompute the comparison unless a quick-demo flag is explicitly enabled.

## 13. W&B plan

### Direct PPO training

- group: `training/direct_ppo/<task_shape>`
- name: `<run_name>`
- tags: `training`, `direct_ppo`, `task:<shape>`, `seed:<seed>`
- config: resolved run config, task selection, full `ppo` config, artifact paths
- summary: final tracking metrics, action metrics, failure modes, model path, evaluation pointers

### Evaluation

- group: `evaluation/<evaluation_name>`
- name: `<source_run_name>__<evaluation_name>`
- tags: `evaluation`, `task:<shape>`, `suite:<evaluation_name>`
- config: evaluation suite, model path, source run manifest, artifact options
- summary: per-task tracking metrics and completion/failure fields

### Curriculum

- group: `curriculum/<curriculum_name>/seed<seed>`
- stage name: `<curriculum_name>__stageNN_<stage_name>`
- tags: `curriculum`, `manual` or `llm`, `stage:NN`, `task:<shape>`
- config: curriculum config, stage task, previous model path, PPO config
- summary: stage metrics, readiness feedback, accepted/rejected proposal counts for LLM

### Comparison

- group: `comparison/<comparison_name>`
- name: `<comparison_name>`
- tags: `comparison`, `final_report`
- config: input run manifests, evaluation suite, aggregation options
- summary: headline final benchmark metrics and winner/ordering fields where meaningful

### Artifact logging

Log small JSON manifests, CSV summaries, plots, and curated GIFs. Do not log large raw traces by default unless explicitly configured.

## 14. Notebook/report contract

The final notebook should load saved artifacts by default and should not require retraining.

Required notebook flags:

```python
TRAIN_FROM_SCRATCH = False
RUN_QUICK_DEMO = True
USE_SAVED_RESULTS = True
```

Expected loaded files:

- `storage/runs/<direct_run>/run_manifest.json`
- `storage/runs/<manual_curriculum_run>/run_manifest.json`
- `storage/runs/<llm_curriculum_run>/run_manifest.json`
- `storage/runs/<comparison_run>/comparison/metrics/comparison_summary.json`
- `storage/runs/<comparison_run>/comparison/metrics/comparison_rows.csv`
- selected plots under `comparison/plots/`
- selected GIFs under each run's `evaluations/<evaluation_name>/renders/`
- copied config files under each run's `config/`

Notebook sections should cover:

- motivation and research question
- environment, observations, actions, and reward
- deterministic task validation
- PPO config and training protocol
- manual curriculum design
- LLM curriculum design and safety boundary
- evaluation protocol and benchmark suites
- quantitative results
- traces, plots, and GIFs
- discussion, limitations, and future work

Main detailed documentation should live in the notebook. The README should be updated at the end to reflect what was actually implemented and include final result GIFs or links to curated GIFs.

## 15. Tests required

### PPO config

- load nested `ppo:` config into a validated config object
- reject invalid learning rates, gamma, batch size, `n_steps`, and total timestep combinations
- verify `PPO(...)` receives config values rather than hidden defaults
- verify metrics, manifests, and W&B config include full resolved PPO config

### Artifact paths

- `storage/runs/<run_name>` helper tests
- normal run directory creation tests
- curriculum stage directory creation tests
- compatibility tests for old `training_runs` and `evaluation_runs` only during migration
- invalid run names and path traversal tests

### Training runner

- config loading and overrides
- task selection by index and shape
- model transfer path validation
- training manifest contents
- disabled post-training evaluation behavior
- minimal monkeypatched PPO training to avoid long jobs

### Policy evaluation

- one-model one-task evaluation writes diagnostics, trace, plots, renders, metrics, manifest
- evaluation can run from a run manifest
- missing model/task failures are clear
- render disabled and plots disabled paths are recorded correctly

### Evaluation suites

- suite config loads and validates tasks
- suite output names are deterministic
- suite can evaluate multiple tasks against one model
- old curriculum benchmark config migration or adapter is tested

### Curriculum training

- manual stage config validation
- sequential model transfer
- per-stage artifact layout
- final model pointer is last stage model
- no top-level final evaluation duplication

### Curriculum evaluation

- own-stage evaluation writes under each stage
- final benchmark evaluation writes under the final stage when requested
- model-scope selection behaves as documented
- baseline/direct comparison input is not duplicated into curriculum stages

### LLM task schema/parser

- schema includes all supported final task fields
- parser accepts JSON object only and rejects markdown/prose/code
- invalid tasks are logged with validation reasons
- repair attempts are bounded and logged
- offline/mock LLM client produces deterministic tests

### Comparison pipeline

- compare direct/manual/LLM run manifests
- aggregate JSON and CSV rows
- write plots from fixture metrics
- missing evaluation artifacts fail clearly
- notebook-ready paths are included in comparison manifest

### CLI help

- `experiments_cli_train_tracking --help`
- `experiments_cli_train_curriculum --help`
- `experiments_cli_evaluate_policy --help`
- `experiments_cli_evaluate_curriculum --help`
- `experiments_cli_compare_runs --help`

### Notebook smoke/nbQA

- run only after the notebook is materially implemented
- verify default flags do not train
- verify saved-result loading cells execute against fixture or small saved artifacts

## 16. Documentation and notebook required

Required:

- update `Drone_RL_LLM_Curriculum.ipynb` as the final report and demonstration artifact
- update `README.md` with final project description, final commands, artifact contract, and result GIFs or links

Optional only after contracts stabilize:

- `docs/run_structure.md`: worth adding only if artifact layout becomes too detailed for README.
- `docs/evaluation_protocol.md`: worth adding only if evaluation suite details need an external protocol reference.
- `docs/llm_curriculum_design.md`: worth adding only if LLM prompts/repair/event logs need a concise appendix.

Avoid creating many separate docs before the final artifact contract is stable. The notebook and README should carry most final documentation.

## 17. Implementation phases

### Phase 0: Completed - current runtime stabilization

- status: completed and validated locally.
- goal: stabilize the current manual curriculum and evaluation pipeline before PPO config extraction or architecture refactoring.
- completed fixes:
  - fixed the manual-curriculum liftoff diagnostic crash in `src/experiments/experiments_ppo_tracking.py`.
  - updated `_task_with_minimum_reference_samples` so diagnostic sample extension prefers extending `duration_sec` or `move_duration_sec` at the existing `sample_rate_hz` instead of increasing `sample_rate_hz`.
  - preserved start-hold metadata during diagnostic task preparation.
  - added fallback-with-warning behavior so liftoff diagnostics do not crash an otherwise valid training run if diagnostic task extension fails.
  - fixed misleading multi-episode evaluation plots by making reset behavior explicit.
  - aligned main report plots with the same single render rollout used for `scenario_rollout.gif` while keeping full multi-episode diagnostics and metrics unchanged.
- files changed during Phase 0:
  - `src/experiments/experiments_ppo_tracking.py`
  - `src/experiments/experiments_policy_evaluation.py`
  - `src/experiments/experiments_curriculum_evaluation.py`
  - `src/evaluation/evaluation_plots.py`
  - `src/evaluation/evaluation_diagnostics.py`
  - `src/validation/validation_tasks.py`
  - `tests/test_experiments_ppo_tracking.py`
  - `tests/test_experiments_policy_evaluation.py`
  - `tests/test_experiments_curriculum_evaluation.py`
  - `tests/test_evaluation_plots.py`
  - `tests/test_envs_task_adapter.py`
  - `tests/test_envs_tracking_env.py`
  - `tests/test_evaluation_diagnostics.py`
- validation completed:
  - `ruff format .`
  - `ruff check .`
  - `mypy src`
  - full `pytest -q`
  - manual curriculum training
  - final-stage `line_basic` benchmark evaluation
- runtime validation result:
  - manual curriculum training finishes without the maximum-acceleration crash.
  - curriculum summary and manifest are written.
  - final-stage benchmark evaluation finishes.
  - full metrics still report repeated truncations where the model fails.
  - main plots and GIF now describe the same render rollout.
- files moved/renamed: none.
- files merged/deleted: none.
- risks closed:
  - liftoff diagnostics no longer invalidate valid acceleration-constrained tasks by changing sample rate.
  - main plots no longer look like multiple stages are overlaid when only multiple reset episodes were present.
- expected commit message: `Stabilize curriculum diagnostics and report plots`.

### Phase 1: PPO config extraction

- goal: remove hardcoded PPO parameters and hidden rollout caps from the training path.
- files likely changed: `configs/training/ppo_tracking.yaml`, `src/experiments/experiments_ppo_tracking.py` initially or new `src/experiments/training/experiments_training_ppo_config.py` if restructuring already happened, `tests/test_experiments_ppo_tracking.py`, `tests/test_utils_wandb.py`.
- files likely moved/renamed: none if done before Phase 2; otherwise later move with the restructuring phase.
- files likely merged/deleted: none.
- tests to run: `pytest tests/test_experiments_ppo_tracking.py tests/test_utils_wandb.py -q` plus a tiny monkeypatched PPO test if available.
- risks: SB3 expects compatible `n_steps` and `batch_size`; small tests need explicit tiny PPO config.
- expected commit message: `Extract PPO hyperparameters into training config`.

### Phase 2: experiments/CLI package restructuring

- goal: split `src/experiments` into subpackages without changing behavior.
- files likely changed: all `src/experiments/*.py` import paths, `src/experiments/__init__.py`, tests importing `src.experiments`, Docker/job docs or commands if module names change.
- files likely moved/renamed: move CLIs to `src/experiments/cli/`, training to `src/experiments/training/`, evaluation to `src/experiments/evaluation/`, curriculum to `src/experiments/curriculum/`, rendering to `src/experiments/rendering/`.
- files likely merged/deleted: retire `cli_training_smoke.py`, `experiments_training_smoke.py`, and `cli_mvp.py` only if equivalent smoke/final commands already exist; otherwise keep temporarily.
- tests to run: `pytest tests/test_experiments_*.py tests/test_cli_mvp.py -q` and CLI help tests.
- risks: import churn and module execution paths can break Docker/HPC usage.
- expected commit message: `Restructure experiment modules into focused subpackages`.

### Phase 3: artifact/run structure

- goal: introduce `storage/runs/<run_name>` as the final artifact contract.
- files likely changed: `src/utils/utils_artifacts.py`, `src/utils/utils_paths.py`, training/evaluation/curriculum/rendering modules, scripts only if explicitly approved, tests for artifact paths.
- files likely moved/renamed: generated artifacts should not be moved automatically.
- files likely merged/deleted: deprecate old category helpers after migration; do not remove immediately.
- tests to run: `pytest tests/test_utils_artifacts.py tests/test_utils_paths.py tests/test_experiments_ppo_tracking.py tests/test_experiments_curriculum_training.py tests/test_experiments_curriculum_evaluation.py -q`.
- risks: breaking existing local artifacts and notebook paths. Use compatibility wrappers during transition.
- expected commit message: `Add unified run artifact layout`.

### Phase 4: policy evaluation suites

- goal: add config-driven evaluation suites for direct, curriculum stage, and final benchmark evaluation.
- files likely changed: `configs/evaluation/*.yaml`, `src/experiments/evaluation/experiments_evaluation_suites.py`, `src/experiments/evaluation/experiments_evaluation_policy.py`, evaluation CLI tests.
- files likely moved/renamed: migrate `configs/evaluation/curriculum_benchmarks.yaml` or support it through an adapter.
- files likely merged/deleted: move benchmark loading out of curriculum evaluation if a shared suite loader covers it.
- tests to run: `pytest tests/test_experiments_policy_evaluation.py tests/test_experiments_curriculum_evaluation.py -q` plus new evaluation suite tests.
- risks: output naming must stay deterministic for notebook/report loading.
- expected commit message: `Add config-driven policy evaluation suites`.

### Phase 5: curriculum as run container

- goal: store manual curriculum runs as one run with per-stage training and evaluation directories.
- files likely changed: `src/experiments/curriculum/experiments_curriculum_manual.py`, curriculum config, artifact helpers, curriculum tests.
- files likely moved/renamed: old curriculum artifacts under `training_runs/curricula` are not moved automatically.
- files likely merged/deleted: merge duplicated stage/benchmark config writing into shared helpers.
- tests to run: `pytest tests/test_experiments_curriculum_training.py tests/test_experiments_curriculum_evaluation.py -q`.
- risks: existing tests expect `training_runs/curricula/...`; update in one controlled pass.
- expected commit message: `Store curriculum stages in unified run containers`.

### Phase 6: comparison pipeline

- goal: compare direct PPO, manual curriculum, and later LLM curriculum through one report-ready pipeline.
- files likely changed: `src/experiments/comparison/experiments_comparison_runs.py`, `src/experiments/comparison/experiments_comparison_plots.py`, `src/experiments/cli/experiments_cli_compare_runs.py`, `configs/comparison/*.yaml`, tests.
- files likely moved/renamed: none.
- files likely merged/deleted: retire `cli_mvp.py` after comparison CLI plus notebook smoke path replaces it.
- tests to run: new comparison tests plus `pytest tests/test_evaluation_plots.py -q`.
- risks: comparing runs with different evaluation suites produces misleading results; enforce suite identity in manifests.
- expected commit message: `Add final run comparison pipeline`.

### Phase 7: LLM curriculum

- goal: implement LLM-guided task proposal, validation, repair, logging, and stage integration.
- files likely changed: `src/llm/*.py`, `src/experiments/curriculum/experiments_curriculum_llm.py`, configs for LLM curriculum, tests.
- files likely moved/renamed: none beyond prior structure.
- files likely merged/deleted: none.
- tests to run: `pytest tests/test_llm_task_schema.py` plus new LLM parser/repair/curriculum tests.
- risks: nondeterministic LLM output, prompt drift, invalid tasks, accidental code/prose output. Use offline/mock tests and deterministic validation.
- expected commit message: `Add validated LLM curriculum proposal pipeline`.

### Phase 8: notebook/report integration

- goal: turn the notebook into the final report that loads saved results by default.
- files likely changed: `Drone_RL_LLM_Curriculum.ipynb`, possibly curated `docs/media/` or `docs/figures/` assets.
- files likely moved/renamed: none unless curated assets are copied into docs.
- files likely merged/deleted: none.
- tests to run: `nbqa ruff Drone_RL_LLM_Curriculum.ipynb` and a quick notebook smoke only after saved fixtures exist.
- risks: notebook execution can become slow or artifact-path brittle. Keep training disabled by default.
- expected commit message: `Integrate final notebook with saved run artifacts`.

### Phase 9: README and final cleanup

- goal: update README and remove obsolete MVP/smoke pathways that are no longer needed.
- files likely changed: `README.md`, final docs/media links, package docs, stale tests.
- files likely moved/renamed: remove or archive obsolete smoke CLIs only after replacement commands exist.
- files likely merged/deleted: delete `cli_mvp.py`, `cli_training_smoke.py`, and `experiments_training_smoke.py` if they are no longer used; keep render smoke only if it remains a CI integration check.
- tests to run: `ruff check .`, `ruff format --check .`, `mypy src`, `pytest -q`.
- risks: deleting smoke helpers too early can remove fast confidence checks. Keep at least one quick command path.
- expected commit message: `Update README and remove obsolete MVP helpers`.

## 18. Do-not-do-now list

- Do not migrate all storage during PPO config extraction; storage layout changes are Phase 3.
- Do not reopen Phase 0 fixes unless a regression appears in curriculum training, evaluation diagnostics, or report plots.
- Do not implement LLM curriculum before manual curriculum and evaluation suites are stable.
- Do not rewrite the notebook until artifact contracts are stable.
- Do not split `src/experiments` before deciding which smoke/prototype files are obsolete or mergeable.
- Do not create many docs before the final artifact contract is stable.
- Do not rename the top-level `src/experiments` package unless a later explicit decision overrides this plan.
- Do not combine PPO config extraction with file moves.
- Do not add figure-eight/star/spiral trajectories unless a final evaluation suite requires them.
- Do not change Docker scripts as part of a pure Python refactor unless the CLI/module paths require it and the change is explicitly scoped.
- Do not edit `Drone_RL_LLM_Curriculum.ipynb` until saved-result paths and comparison outputs are defined.

## 19. Immediate next actions

1. Commit the completed Phase 0 stabilization checkpoint if it has not been committed yet. Do not include generated `storage/` artifacts.

2. Start Phase 1 by adding explicit PPO configuration tests. The first tests should require a nested `ppo:` block with values for `policy`, `device`, `learning_rate`, `gamma`, `gae_lambda`, `n_steps`, `batch_size`, `n_epochs`, `clip_range`, `ent_coef`, `vf_coef`, `max_grad_norm`, and `target_kl`.

3. Implement a focused PPO config loader/validator before restructuring packages. Prefer a small current-location module such as `src/experiments/experiments_ppo_config.py` for now. Do not move it into `src/experiments/training/` until Phase 2.

4. Update `src/experiments/experiments_ppo_tracking.py` so `PPO(...)` receives its hyperparameters from the resolved config instead of hardcoded constructor values or hidden rollout-step caps.

5. Validate Phase 1 with:

```bash
ruff format .
ruff check .
mypy src
pytest -q tests/test_experiments_ppo_config.py tests/test_experiments_ppo_tracking.py tests/test_utils_wandb.py
pytest -q

python -m src.experiments.cli_train_tracking \
  --config configs/training/ppo_tracking.yaml \
  --task-shape line \
  --run-name local_line_ppo_config_smoke \
  --total-timesteps 512 \
  --eval-steps 40 \
  --seed 0 \
  --wandb-mode disabled