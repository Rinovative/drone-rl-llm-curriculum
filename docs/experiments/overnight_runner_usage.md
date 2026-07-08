# Overnight Medium-Screening Lanes

This matrix is medium-screening seed0 only. Final-budget and multi-seed runs are intentionally deferred until after these results are reviewed.

The matrix remains 18 experiments. Unit accounting is now direct PPO = 1, manual curriculum = 5, and LLM curriculum = 10, for 44 total units and 11 units per lane.

Manual curriculum runs use five fixed-budget stages for interpretability: hover stabilization, vertical low/high stabilization, start-hold short line, L-shaped polyline tracking, and the medium tracking-distribution stage.

LLM curriculum runs use ten adaptive budget stages. The LLM may choose only these bounded profiles:

| profile | stage timesteps | intended use |
| --- | ---: | --- |
| bootstrap | 750000 | Stage 1 policy warmup, 1.5x reference medium |
| short | 375000 | Easy confirmation stages, 0.75x reference medium |
| normal | 500000 | Ordinary progression, 1.0x reference medium |
| recovery | 625000 | Previous stage unstable but promising, 1.25x reference medium |
| extend | 750000 | Use sparingly when appropriate but undertrained, 1.5x reference medium |

The LLM total stage-budget cap is 5500000 timesteps, which is 11x the 500000-step reference medium config. Stage 1 defaults to `bootstrap`; later stages may use `short`, `normal`, `recovery`, or `extend`. The resolver reserves enough short-stage budget for all remaining stage slots, falls back to the largest safe profile that still fits when a requested profile would exceed the cap, and logs every clipping or fallback decision in proposal logs and stage/run summaries.

Start all four lanes from `/workspace/repo` in four terminals:

```bash
export LANE_RUN_ID="$(date +%Y%m%d_%H%M%S)"
bash scripts/run_lane_1.sh
bash scripts/run_lane_2.sh
bash scripts/run_lane_3.sh
bash scripts/run_lane_4.sh
```

To disable W&B for every training command in a lane, prefix the command explicitly:

```bash
WANDB_MODE_OVERRIDE=disabled bash scripts/run_lane_1.sh
```

The lane scripts do not disable W&B internally; they use config-level W&B settings unless `WANDB_MODE_OVERRIDE` is set by the operator.

Logs and markers are written under `storage/logs/overnight_lanes/<LANE_RUN_ID>/lane_<N>/`. Each experiment gets train, standard evaluation, variation evaluation, broad evaluation, and render logs. Lane summaries are `lane_summary.tsv` and `lane_summary.md` in the same lane log directory.

If a run manifest already exists at `storage/runs/<run_name>/run_manifest.json`, training is skipped and evaluation/render phases still run unless their marker files already exist. Re-run a lane with the same `LANE_RUN_ID` to resume from markers.

LLM curriculum lanes do not start a local LLM server. They preflight the configured OpenAI-compatible `/models` endpoint and mark only that LLM experiment as skipped if it is unreachable.

Standard evaluation uses `src.experiments.cli.experiments_cli_evaluate_policy` for direct PPO and `src.experiments.cli.experiments_cli_evaluate_curriculum --model-scope final-stage` for curricula. Variation and broad suites use `scripts/evaluate_variation_suite.sh` with `configs/evaluation/evaluation_task_suite_variation.yaml` and `configs/evaluation/evaluation_task_suite_broad.yaml`. Render status is reported by `scripts/render_run_gifs.sh`; GIFs are produced by evaluation suites when simulator rendering succeeds.

Curricula are evaluated only once after the full curriculum completes, using the final-stage model. Full variation, broad, and render phases are not run for every individual LLM stage.

Task-distribution training lanes use only `configs/tasks/task_distribution_tracking_medium.yaml`. Small and broad task-distribution configs remain for debugging/evaluation/future work, but they are not scheduled tonight.
