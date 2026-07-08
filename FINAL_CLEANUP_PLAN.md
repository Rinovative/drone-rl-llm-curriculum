# Final Cleanup Roadmap

This document is the current roadmap for the drone RL with LLM-guided curriculum repository. It reflects the completed Phase 0 through Phase 3 work, the post-audit cleanup status, and the next planned phases. It is documentation only; it does not start Phase 4, edit notebooks, migrate storage, or change training, reward, action, or evaluation semantics.

## 1. Current Status

The repository now has the intended source-package split and canonical run layout:

- Core reusable logic lives in `src/envs`, `src/trajectories`, `src/validation`, `src/evaluation`, `src/llm`, and `src/utils`.
- Experiment orchestration lives under `src/experiments` with focused subpackages for CLI entry points, training, evaluation, curriculum, and rendering.
- Generated run artifacts use the canonical `storage/runs/<self_describing_run_id>` layout through `src/utils/utils_artifacts.py`.
- The old root-level experiment modules and CLI wrappers are intentionally removed. Tests keep negative coverage so those import paths stay removed.
- PPO hyperparameters are configured through the nested `ppo:` block in `configs/training/ppo_tracking.yaml`.
- The post-audit legacy cleanup is complete: flat PPO keys, mixed flat+nested PPO configs, old root experiment imports, and stale Docker job references to removed CLI paths are rejected or absent from active code.

Phase 3 is completed. The next real implementation phase is Phase 4: config-driven policy evaluation suites. This cleanup does not implement Phase 4.

## 2. Current Source Layout

```text
src/
├── envs/           # Drone environment builders, task adapters, wrappers, rewards
├── trajectories/   # Trajectory primitives and reference path utilities
├── validation/     # Deterministic task contracts and feasibility checks
├── evaluation/     # Reusable metrics, rollout traces, diagnostics, and plots
├── llm/            # LLM task schema helpers; full curriculum proposal logic is future work
├── utils/          # Paths, canonical artifacts, W&B helpers
└── experiments/    # Experiment orchestration and CLI entry points
    ├── cli/        # python -m entry-point modules
    ├── training/   # PPO config, PPO tracking training, deterministic training smoke
    ├── evaluation/ # Policy evaluation orchestration
    ├── curriculum/ # Manual curriculum training/evaluation and curriculum validation summaries
    └── rendering/  # Policy, scenario, and render-smoke workflows
```

The current canonical experiment imports use the new subpackages directly, for example:

```python
from src.experiments.training import experiments_training_ppo_config as ppo_config
from src.experiments.training import experiments_training_ppo_tracking as ppo_tracking
from src.experiments.evaluation import experiments_evaluation_policy as policy_evaluation
from src.experiments.curriculum import experiments_curriculum_training as curriculum_training
from src.experiments.rendering import experiments_rendering_policy as policy_render
```

The root `src.experiments` package exposes static subpackage aliases only. It does not expose compatibility aliases for old implementation modules.

## 3. Canonical Artifact Layout

The canonical generated artifact contract is run-scoped:

```text
storage/runs/<self_describing_run_id>/
├── run_manifest.json
├── config/
│   └── evaluation_suites/
├── training/
│   ├── manifest.json
│   ├── models/
│   ├── metrics/
│   ├── diagnostics/
│   ├── logs/
│   └── wandb/
├── evaluations/
│   └── <evaluation_name>/
│       ├── diagnostics/
│       ├── traces/
│       ├── plots/
│       ├── renders/
│       ├── metrics/
│       └── manifests/
└── stages/
    └── stageNN_<stage_name>/
        ├── training/
        └── evaluations/
```

Direct PPO, manual curriculum, rendering, W&B, and evaluation helpers should use this run container contract by default. Generated artifacts should not be committed. Existing generated artifacts should not be bulk-moved by cleanup work.

The intentionally removed legacy directories remain removed from active code:

- `storage/training_runs`
- `storage/evaluation_runs`
- `storage/comparison_reports`

## 4. Completed Phases

### Phase 0: Completed - runtime stabilization

Phase 0 stabilized the existing manual curriculum and evaluation flow before architecture changes. The key result was that manual curriculum training and final-stage benchmark evaluation could complete without the previous liftoff-diagnostic crash. It also aligned report plots with the same single render rollout used for the showcase GIF while preserving full diagnostic metrics.

### Phase 1: Completed - PPO config extraction

Phase 1 moved PPO hyperparameters out of hardcoded constructor arguments and into explicit config handling. Current `configs/training/ppo_tracking.yaml` includes a nested `ppo:` block with the resolved Stable-Baselines3 PPO settings. The training path passes those values into `PPO(...)`, records the resolved PPO config in metrics/manifests/W&B config, and validates tiny smoke budgets against explicit `ppo.n_steps`.

The post-audit cleanup tightened the Phase 1 contract: missing `ppo:`, legacy flat/top-level PPO keys, and mixed flat+nested PPO configs are no longer accepted by the training settings loader.

### Phase 2: Completed - experiments package restructuring

Phase 2 split the former flat `src/experiments` package into static subpackages:

- `src/experiments/cli`
- `src/experiments/training`
- `src/experiments/evaluation`
- `src/experiments/curriculum`
- `src/experiments/rendering`

Old root-level CLI and implementation module paths were removed as a clean breaking migration. Negative tests intentionally assert that the old imports fail.

### Phase 3: Completed - canonical run artifact layout

Phase 3 introduced `storage/runs/<self_describing_run_id>` as the canonical flat run container. `src/utils/utils_artifacts.py` owns the canonical path helpers for training, evaluation, curriculum stages, manifests, metrics, renders, traces, diagnostics, logs, models, and W&B output.

Phase 3 is not open. Do not re-migrate storage or reintroduce old storage helpers as compatibility layers.

## 5. Post-Audit Static Analysis And Legacy Findings

Post-audit static analysis found no broad source breakage, and the follow-up legacy cleanup leaves no active compatibility paths for the removed experiment modules or flat PPO config form:

- `ruff check .` passed during the audit.
- `mypy src` passed during the audit.
- `pytest -q` passed during the audit.
- `pyright` and `python -m pyright` were unavailable in the environment, so exact Pyright CLI output could not be collected without adding a new tool.
- `.vscode/settings.json` points Pylance at `/opt/venv/bin/python` with `python.analysis.typeCheckingMode` set to `basic`.

Cleanup validation for this handoff passed with `bash -n scripts/docker_job.sh`, `ruff check .`, `ruff format --check .`, `mypy src`, the focused PPO/package cleanup pytest set, and `pytest -q`.

Likely Pylance noise classes after the restructure are stale editor cache, wrong workspace root, old deleted module paths, test-only private helper access, or third-party missing-stub reports. Canonical package imports resolve through the configured interpreter.

Legacy scan status:

- Old root experiment module references remain only in negative tests and historical cleanup documentation.
- `__getattr__` compatibility aliases are absent from `src.experiments`.
- `artifact_layout` remains only in tests that assert it is absent from metrics/manifests.
- `scripts/docker_job.sh` usage points at `src/experiments/cli/experiments_cli_train_tracking.py` and no longer advertises `src/experiments/cli_train_tracking.py`.
- The flat PPO-key compatibility path is removed; tests cover missing, flat, and mixed PPO config forms at both helper and training-settings load boundaries.

## 6. Remaining Cleanup

These are cleanup items, not Phase 4 implementation:

- Keep old-root-module negative tests in `tests/test_experiments_package_structure.py`.
- Keep reviewing user-facing smoke/MVP wording in docstrings and README before final report polish.
- Keep explicit output-dir override behavior unless a later scoped task removes category-root overrides.
- Do not make private rendering helpers public only to satisfy tests; expose public helpers only when an implementation phase needs a stable interface.
- Do not add compatibility wrappers, old import aliases, optional old/new behavior, or old storage directories.

## 7. Future Roadmap

### Phase 4: Policy evaluation suites
Phase 4 must use the canonical `storage/runs` layout and must not add adapters for old benchmark configs. Evaluation suites should become the canonical source for benchmark tasks.
Add config-driven evaluation suites for direct PPO, curriculum stages, and final benchmark evaluation. This should likely add a suite loader under `src/experiments/evaluation`, update evaluation CLI coverage, and keep output naming deterministic for notebook/report loading.

### Phase 5: Comparison pipeline

Add a report-ready comparison workflow for direct PPO, manual curriculum, and later LLM curriculum runs. Expected outputs include JSON summaries, CSV rows, plots, and a manifest that records suite identity so mismatched evaluations cannot be compared silently.

### Phase 6: LLM curriculum

Implement the LLM-guided proposal pipeline with strict JSON parsing, deterministic validation, bounded repair, event logging, and offline/mock tests. The LLM remains a curriculum generator only; it must not control the drone or generate executable Python code.

### Phase 7: Notebook and report integration

Turn `Drone_RL_LLM_Curriculum.ipynb` into the final report and demo artifact after saved-result paths and comparison outputs are stable. The notebook should load saved results by default, avoid retraining by default, and import reusable logic from `src/`.

### Phase 8: README and final cleanup

Update README commands, artifact descriptions, and final workflow documentation after the evaluation, comparison, LLM curriculum, and notebook contracts are stable. Retire obsolete MVP/smoke helpers only after replacement quick checks exist.

## 8. Do-Not-Change List

- Do not create commits during Codex tasks.
- Do not start Phase 4 as part of cleanup.
- Do not edit `Drone_RL_LLM_Curriculum.ipynb` until notebook/report integration is explicitly requested.
- Do not migrate storage again or bulk-move generated artifacts.
- Do not change PPO hyperparameter values or PPO training semantics.
- Do not change reward logic.
- Do not change action semantics.
- Do not change evaluation metrics.
- Do not add legacy compatibility layers, wrappers, aliases, fallback APIs, or optional old/new behavior.
- Do not reintroduce old root-level experiment modules.
- Do not reintroduce `storage/training_runs`, `storage/evaluation_runs`, or `storage/comparison_reports`.

## 9. Validation Baseline

For this post-audit cleanup, run:

```bash
bash -n scripts/docker_job.sh
ruff check .
ruff format --check .
mypy src
pytest tests/test_experiments_ppo_config.py tests/test_experiments_package_structure.py -q
pytest -q
```
