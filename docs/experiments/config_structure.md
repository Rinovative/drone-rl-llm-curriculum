# Config Structure Source of Truth

This repository keeps active experiment configs intentionally small for the current 18-experiment, 6-lane medium-screening matrix. Runtime outputs, W&B data, models, videos, caches, and generated run artifacts belong under `storage/`, not in the config tree.

## Active Training Entry Points

Direct PPO lane configs live in `configs/training/`. Every direct config separates the deterministic own-task representative from the actual training distribution:

- `task_config_path`: `configs/training/ppo_tracking_representative_tasks.yaml`
- `task_index`: deterministic representative/own-task selector
- `task_distribution_config_path`: actual per-episode training distribution when the run is distribution-based

Current direct PPO configs are:

- `ppo_tracking_pid_dynprev_basic_show.yaml`
- `ppo_tracking_directrpm_dynprev_basic_show.yaml`
- `ppo_tracking_pid_dynprev_m-taskdist_medium.yaml`
- `ppo_tracking_directrpm_dynprev_m-taskdist_medium.yaml`
- `ppo_tracking_pid_dynprev_net256_m-taskdist_medium.yaml`
- `ppo_tracking_directrpm_dynprev_net256_m-taskdist_medium.yaml`
- `ppo_tracking_pid_dynprev_m-taskdist_medium_low_lr.yaml`
- `ppo_tracking_directrpm_dynprev_m-taskdist_medium_low_lr.yaml`
- `ppo_tracking_pid_dynprev_m-taskdist_medium_ent005.yaml`
- `ppo_tracking_directrpm_dynprev_m-taskdist_medium_ent005.yaml`
- `ppo_tracking_pid_dynprev_m-taskdist_medium_clip010.yaml`
- `ppo_tracking_directrpm_dynprev_m-taskdist_medium_clip010.yaml`
- `ppo_tracking_pid_dynprev_m-taskdist_medium_targetkl015.yaml`
- `ppo_tracking_directrpm_dynprev_m-taskdist_medium_targetkl015.yaml`

Net128 is the default architecture for normal runs. Net256 appears only as the paired architecture comparison. Net512 and the old no-dynamics PID baseline are not scheduled.

For `pid_position`, PPO still sees normalized actions in [-1, 1]. Action dim 2 is a normalized PID z target, with active PID configs using `pid_target_z_min_m: 0.2` and `pid_target_z_max_m: 1.5` so altitude-changing references above 1.0 m remain reachable with margin. For `direct_rpm`, action dim 2 remains motor 2 command; PID z reachability metadata is not part of the direct-RPM action semantics.

Manual and LLM curriculum lane configs live in `configs/curricula/`:

- `curriculum_pid_dynprev_m-taskdist_medium.yaml`
- `curriculum_directrpm_dynprev_m-taskdist_medium.yaml`
- `llm_curriculum_pid_dynprev_m-taskdist_medium.yaml`
- `llm_curriculum_directrpm_dynprev_m-taskdist_medium.yaml`

Manual curriculum uses five focused stage distributions: hover, vertical, start-hold short line, polyline/L-shape, and tracking medium. The pure vertical stage is an early altitude-control diagnostic/training step, then the manual sequence moves into XY and turn-following stages.

LLM curriculum uses the hover bootstrap distribution for Stage 1, then accepts only validated bounded task distributions for later stages. Progression roles are explicit: hover and pure vertical are bootstrap/recovery only, short-line/line tasks are early XY progression, angled-vertical/delayed-altitude/multi-height tasks combine XY and z, polyline/zigzag/triangle/rectangle tasks cover turns, circle/ellipse tasks cover gentle curvature, `tracking_small` is fallback/consolidation, and `tracking_medium` is late/final broad coverage. Deterministic proposal repair prevents hover/vertical loops, repeated related families, and early broad proposals; stage summaries record `proposal_repaired`, `proposal_repair_reason`, original/final distribution IDs, the applied progression rule, `hover_vertical_loop_detected`, and `stage_progression_bucket`.

## Task Sources

Training distributions live in `configs/tasks/` and carry these source-of-truth fields:

- `task_config_role: training_distribution`
- `task_is_distribution: true`
- `sampled_per_episode: true`
- `constant_within_episode: true`
- `variation_enabled: true`

Scheduled direct PPO training uses:

- `task_distribution_basic_training_show.yaml`
- `task_distribution_tracking_medium.yaml`

Curriculum and LLM support uses:

- `task_distribution_hover_bootstrap_medium.yaml`
- `task_distribution_vertical_bootstrap_medium.yaml`
- `task_distribution_vertical_up_down_bootstrap_medium.yaml`
- `task_distribution_angled_vertical_bootstrap_medium.yaml`
- `task_distribution_delayed_altitude_polyline_bootstrap_medium.yaml`
- `task_distribution_short_line_bootstrap_medium.yaml`
- `task_distribution_polyline_bootstrap_medium.yaml`
- `task_distribution_tracking_medium.yaml`
- `task_distribution_multi_height_polyline_bootstrap_medium.yaml`
- `task_distribution_triangle_bootstrap_medium.yaml`
- `task_distribution_zigzag_bootstrap_medium.yaml`
- `task_distribution_rectangle_bootstrap_medium.yaml`
- `task_distribution_circle_bootstrap_medium.yaml`
- `task_distribution_ellipse_bootstrap_medium.yaml`

`task_distribution_hover_small.yaml`, `task_distribution_line_small.yaml`, `task_distribution_tracking_small.yaml`, and `task_distribution_tracking_broad.yaml` are retained as tested LLM-schema/compatibility support. They are not scheduled by the 18-experiment matrix.

Active training, curriculum, representative, generalization, and scenario/show references use the standard base reference height around z=1.0 m. Level tasks use fixed z=1.0 or sampled base_z_range_m [0.9, 1.1], while altitude-control tasks stay anchored near 1.0 m and may climb or descend within validation limits. All active categories use start_hold_sec: 1.0 with start holds excluded from tracking metrics, and the held point remains the first moving tracking point so no reference discontinuity is introduced.

Active PPO training configs set `initial_state.mode: reference_start`, so `TrajectoryTrackingEnv` passes the first reference position as `HoverAviary.initial_xyzs` and refreshes `base_env.INIT_XYZS` after each sampled task-distribution reset. Manifests and traces record `initial_state_mode`, requested and actual initial XYZ, the initial reference XYZ, and spawn/reference error fields so any return to near-ground default spawning is visible.

The tracking reward is not masked during start hold. Task metadata records `start_hold_reward_policy: full_tracking_reward_active_during_uniform_reference_start_hold` and `tracking_reward_starts_after_start_hold: false`. Later altitude learning remains active through takeoff/vertical tasks, vertical up/down, angled climb/descent, delayed-altitude polylines, and multi-height polylines.

Where validation margins allowed it, line, polyline, zigzag, triangle, rectangle, square, circle, ellipse, figure-eight, multi-height, and show paths were enlarged moderately and durations were raised with the geometry so reference speeds remain conservative. The basic and standard shows now begin with easier horizontal/diagonal XY motion; vertical or altitude-changing movement appears later after the start/settle hold and at least one easier moving segment.

## Representative Evaluation Tasks

`configs/training/ppo_tracking_representative_tasks.yaml` is representative-only. It replaced the old ambiguous `ppo_tracking_tasks.yaml` name. It is used for deterministic own-task/evaluation selection and for documenting the representative counterpart of each training distribution; it is not an actual varied training source.

Representative entries intentionally avoid training-decision metadata such as sampling ranges, LLM proposal fields, and accepted/repaired task fields. The actual training distribution is always read from a `task_distribution_config_path` or curriculum stage distribution path.

## Standard Evaluation

The active standard evaluation structure is:

- own-task evaluation from each run's representative task source
- `configs/evaluation/generalization_eval_suite.yaml`
- `configs/evaluation/scenarios/show_easy.yaml`
- `configs/evaluation/scenarios/show_medium.yaml`
- `configs/evaluation/scenarios/show_hard.yaml`
- render status through `scripts/render_run_gifs.sh`

Scenario configs point at `configs/evaluation/scenario_task_catalog.yaml` and are evaluation/show references only. They are not training distributions, and `task_distribution_basic_training_show.yaml` is not a scenario/show config.

Older broad, variation, line, and final benchmark suites are not active configs. Legacy suite YAMLs used only for parser/regression tests live under `tests/fixtures/configs/evaluation/`.

## Smoke And Test Fixtures

There is no active `configs/smoke/` folder. Tiny smoke and legacy scenario configs were moved to `tests/fixtures/configs/` so unit tests stay fast without making smoke configs look like production lane inputs.

The old `configs/scenarios/` folder was also removed. Active standard scenarios are exactly `show_easy`, `show_medium`, and `show_hard` under `configs/evaluation/scenarios/`.

## Run Manifests

Run and stage manifests distinguish training source from representative source with fields such as:

- `training_config_path`
- `training_task_distribution_config_path`
- `representative_task_config_path`
- `own_task_eval_config_path`
- `generalization_config_path`
- `scenario_config_paths`
- `training_task_is_distribution`
- `sampled_per_episode`
- `variation_enabled`
- `constant_within_episode`
- `training_task_distribution_snapshot`
- `representative_eval_task_snapshot`

Scenario evaluation reconstructs the environment from the direct run manifest or the final curriculum stage manifest, preserves trained action and observation settings including PID z target bounds, swaps only the scenario reference, and records model/env observation-space diagnostics before policy rollout.

PID-position manifests, traces, and diagnostics record z reachability fields such as `pid_target_z_min_m`, `pid_target_z_max_m`, `real_pid_z_target_low`, `real_pid_z_target_high`, `reference_z_min`, `reference_z_max`, `reference_z_reachable_by_pid_position`, and z margin values. Action saturation should be interpreted together with these fields: unreachable z references are action-space/configuration issues, while saturated action_2 with a reachable reference is a policy/control signal.

Direct PPO runs own their detailed evaluation tree at `storage/runs/<direct_run>/evaluations/`. Curriculum runs own detailed own-task, generalization, and scenario outputs inside the relevant `stages/stageXX_<name>/evaluations/` folder. The curriculum root keeps only manifests, compact summaries, and pointers such as `final_stage_evaluation_path`; root-level duplicate `evaluations/own_task`, `evaluations/generalization`, or `evaluations/scenarios` trees are not the source of truth.

## Removed Active Legacy Configs

These names are no longer active configs:

- `configs/training/ppo_tracking_final.yaml`
- `configs/training/ppo_tracking_medium.yaml`
- `configs/training/ppo_tracking_tasks.yaml`
- `configs/training/*smoke*.yaml`
- `configs/curricula/curriculum_manual_line_{smoke,medium,final}.yaml`
- `configs/curricula/curriculum_llm_*.yaml`
- `configs/smoke/`
- `configs/scenarios/`
