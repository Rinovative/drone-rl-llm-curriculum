# Overnight Medium-Screening Lanes

This matrix is medium-screening seed0 only. Final-budget and multi-seed runs are intentionally deferred until after these results are reviewed.

The final matrix remains 18 experiments. Unit accounting is now direct PPO = 1, manual curriculum = 5, and LLM curriculum = 5, for 34 total units across six lanes. The preferred lane totals are 6, 6, 6, 6, 5, and 5 units.

PPO policy architecture is net128 by default for direct, manual, and LLM runs. The only architecture comparison is a paired PID/direct-RPM net256 tracking-medium run. Net512 is not scheduled. `low_lr`, `ent005`, `clip010`, and `targetkl015` are paired PID/direct-RPM PPO-profile comparisons on the net128 default architecture.

Manual curriculum runs use five fixed-budget stages for interpretability: hover stabilization, vertical low/high stabilization, start-hold short line, L-shaped polyline tracking, and the medium tracking-distribution stage. Each manual stage uses the 500000-step reference medium budget, for 2500000 total timesteps.

LLM curriculum runs use ten adaptive budget stages. Stage 1 is a deterministic randomized hover-target stabilization bootstrap; the LLM begins proposing tasks at Stage 2. The LLM may choose only these bounded profiles after deterministic validation:

| profile | stage timesteps | intended use |
| --- | ---: | --- |
| bootstrap | 500000 | Stage 1 randomized hover-target warmup, 1.0x reference medium |
| short | 175000 | Easy confirmation stages, 0.35x reference medium |
| normal | 250000 | Ordinary progression, 0.50x reference medium |
| recovery | 325000 | Previous stage unstable but promising, 0.65x reference medium |
| extend | 400000 | Use sparingly when appropriate but undertrained, 0.80x reference medium |

The LLM total stage-budget cap is 2500000 timesteps, which is 5x the 500000-step reference medium config and comparable to the manual curriculum total budget. Stage 1 uses `bootstrap`; later stages may use `short`, `normal`, `recovery`, or `extend`. The resolver reserves enough short-stage budget for all remaining stage slots, falls back to the largest safe profile that still fits when a requested profile would exceed the cap, and logs every clipping or fallback decision in proposal logs and stage/run summaries.

Start all six lanes from `/workspace/repo` in six terminals:

```bash
export LANE_RUN_ID="$(date +%Y%m%d_%H%M%S)"
bash scripts/run_lane_1.sh
bash scripts/run_lane_2.sh
bash scripts/run_lane_3.sh
bash scripts/run_lane_4.sh
bash scripts/run_lane_5.sh
bash scripts/run_lane_6.sh
```

To disable W&B for every training command in a lane, prefix the command explicitly:

```bash
WANDB_MODE_OVERRIDE=disabled bash scripts/run_lane_1.sh
```

The lane scripts do not disable W&B internally; they use config-level W&B settings unless `WANDB_MODE_OVERRIDE` is set by the operator.

Logs and markers are written under `storage/logs/overnight_lanes/<LANE_RUN_ID>/lane_<N>/`. Each experiment gets train, simplified standard evaluation, and render-status logs. Lane summaries are `lane_summary.tsv` and `lane_summary.md` in the same lane log directory.

If a run manifest already exists at `storage/runs/<run_name>/run_manifest.json`, training is skipped and evaluation/render phases still run unless their marker files already exist. Re-run a lane with the same `LANE_RUN_ID` to resume from markers.

LLM curriculum lanes do not start a local LLM server. They preflight the configured OpenAI-compatible `/models` endpoint and mark only that LLM experiment as skipped if it is unreachable.

Standard evaluation uses `src.experiments.cli.experiments_cli_evaluate_policy` for direct PPO and `src.experiments.cli.experiments_cli_evaluate_curriculum --model-scope final-stage` for curricula. Evaluation prefers a recorded best model when a manifest exposes one and otherwise falls back to the last saved model. Direct PPO runs write detailed outputs at `evaluations/own_task`, `evaluations/generalization`, and `evaluations/scenarios/{easy,medium,hard}`. Curriculum runs write detailed own-task, generalization, and scenario outputs only inside the owning `stages/stageXX_<name>/evaluations/` folder; root curriculum summaries point to those paths and set `root_evaluation_outputs_duplicated: false`. Render status is reported by `scripts/render_run_gifs.sh` under the run evaluation area; GIFs are produced by evaluation suites and scenarios when simulator rendering succeeds.

Curricula are evaluated only once after the full curriculum completes, using the final-stage best model when available and the final-stage last model otherwise. Scenario/show evaluation uses that final-stage model; per-stage scenario renders are not launched by the lane runner. Active training, curriculum, representative, generalization, and scenario/show references use the standard base height around z=1.0 m. Level tasks use z=1.0 or sampled base_z_range_m [0.9, 1.1], altitude-control tasks remain available above and below that anchor, and every active category uses start_hold_sec: 2.0 with start holds excluded from tracking metrics where configured.

Direct basic-show lanes train on `configs/tasks/task_distribution_basic_training_show.yaml`; tracking-medium task-distribution lanes train on `configs/tasks/task_distribution_tracking_medium.yaml`. The basic and standard shows begin with horizontal or diagonal XY motion before vertical/altitude-changing segments, and active shapes are moderately larger where validation margins allow. The start-hold reward remains the full tracking reward, with metadata recording `full_tracking_reward_active_during_uniform_start_hold`. Small and broad task-distribution configs remain only as tested LLM-schema/compatibility support, and they are not scheduled tonight. See `docs/experiments/config_structure.md` for the source-of-truth config map.
