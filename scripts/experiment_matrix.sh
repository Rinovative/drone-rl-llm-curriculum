#!/usr/bin/env bash
# Medium-screening overnight experiment matrix. Source this file from lane runners.

experiment_kind() {
  case "$1" in
    direct_ppo_pid_baseline_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_pid_dynprev_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_pid_dynprev_net128_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_directrpm_dynprev_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_pid_dynprev_taskdist_medium_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_pid_dynprev_net128_taskdist_medium_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_directrpm_dynprev_taskdist_medium_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_directrpm_dynprev_net128_taskdist_medium_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_pid_dynprev_taskdist_medium_low_lr_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_pid_dynprev_taskdist_medium_ent005_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_pid_dynprev_net128_taskdist_medium_low_lr_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_pid_dynprev_net128_taskdist_medium_ent005_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_directrpm_dynprev_taskdist_medium_low_lr_medium_seed0) echo "direct_ppo" ;;
    direct_ppo_directrpm_dynprev_taskdist_medium_ent005_medium_seed0) echo "direct_ppo" ;;
    curriculum_manual_pid_dynprev_taskdist_medium_medium_seed0) echo "manual_curriculum" ;;
    curriculum_manual_directrpm_dynprev_taskdist_medium_medium_seed0) echo "manual_curriculum" ;;
    curriculum_llm_local_pid_dynprev_taskdist_medium_medium_seed0) echo "llm_curriculum" ;;
    curriculum_llm_local_directrpm_dynprev_taskdist_medium_medium_seed0) echo "llm_curriculum" ;;
    *) return 1 ;;
  esac
}

experiment_config() {
  case "$1" in
    direct_ppo_pid_baseline_medium_seed0) echo "configs/training/ppo_tracking_pid_baseline_medium.yaml" ;;
    direct_ppo_pid_dynprev_medium_seed0) echo "configs/training/ppo_tracking_pid_dynprev_medium.yaml" ;;
    direct_ppo_pid_dynprev_net128_medium_seed0) echo "configs/training/ppo_tracking_pid_dynprev_net128_medium.yaml" ;;
    direct_ppo_directrpm_dynprev_medium_seed0) echo "configs/training/ppo_tracking_directrpm_dynprev_medium.yaml" ;;
    direct_ppo_pid_dynprev_taskdist_medium_medium_seed0) echo "configs/training/ppo_tracking_pid_dynprev_taskdist_medium_medium.yaml" ;;
    direct_ppo_pid_dynprev_net128_taskdist_medium_medium_seed0) echo "configs/training/ppo_tracking_pid_dynprev_net128_taskdist_medium_medium.yaml" ;;
    direct_ppo_directrpm_dynprev_taskdist_medium_medium_seed0) echo "configs/training/ppo_tracking_directrpm_dynprev_taskdist_medium_medium.yaml" ;;
    direct_ppo_directrpm_dynprev_net128_taskdist_medium_medium_seed0) echo "configs/training/ppo_tracking_directrpm_dynprev_net128_taskdist_medium_medium.yaml" ;;
    direct_ppo_pid_dynprev_taskdist_medium_low_lr_medium_seed0) echo "configs/training/ppo_tracking_pid_dynprev_taskdist_medium_low_lr_medium.yaml" ;;
    direct_ppo_pid_dynprev_taskdist_medium_ent005_medium_seed0) echo "configs/training/ppo_tracking_pid_dynprev_taskdist_medium_ent005_medium.yaml" ;;
    direct_ppo_pid_dynprev_net128_taskdist_medium_low_lr_medium_seed0) echo "configs/training/ppo_tracking_pid_dynprev_net128_taskdist_medium_low_lr_medium.yaml" ;;
    direct_ppo_pid_dynprev_net128_taskdist_medium_ent005_medium_seed0) echo "configs/training/ppo_tracking_pid_dynprev_net128_taskdist_medium_ent005_medium.yaml" ;;
    direct_ppo_directrpm_dynprev_taskdist_medium_low_lr_medium_seed0) echo "configs/training/ppo_tracking_directrpm_dynprev_taskdist_medium_low_lr_medium.yaml" ;;
    direct_ppo_directrpm_dynprev_taskdist_medium_ent005_medium_seed0) echo "configs/training/ppo_tracking_directrpm_dynprev_taskdist_medium_ent005_medium.yaml" ;;
    curriculum_manual_pid_dynprev_taskdist_medium_medium_seed0) echo "configs/curricula/curriculum_manual_pid_dynprev_taskdist_medium_medium.yaml" ;;
    curriculum_manual_directrpm_dynprev_taskdist_medium_medium_seed0) echo "configs/curricula/curriculum_manual_directrpm_dynprev_taskdist_medium_medium.yaml" ;;
    curriculum_llm_local_pid_dynprev_taskdist_medium_medium_seed0) echo "configs/curricula/curriculum_llm_local_pid_dynprev_taskdist_medium_medium.yaml" ;;
    curriculum_llm_local_directrpm_dynprev_taskdist_medium_medium_seed0) echo "configs/curricula/curriculum_llm_local_directrpm_dynprev_taskdist_medium_medium.yaml" ;;
    *) return 1 ;;
  esac
}

experiment_run_name() { echo "$1"; }

experiment_units() {
  case "$1" in
    curriculum_manual_*) echo "5" ;;
    curriculum_llm_*) echo "10" ;;
    direct_ppo_*) echo "1" ;;
    *) return 1 ;;
  esac
}

experiment_priority() {
  case "$1" in
    direct_ppo_directrpm_dynprev_net128_taskdist_medium_medium_seed0|direct_ppo_directrpm_dynprev_taskdist_medium_low_lr_medium_seed0|direct_ppo_directrpm_dynprev_taskdist_medium_ent005_medium_seed0|curriculum_manual_directrpm_dynprev_taskdist_medium_medium_seed0) echo "should-have" ;;
    curriculum_llm_local_directrpm_dynprev_taskdist_medium_medium_seed0) echo "optional" ;;
    *) echo "must-have" ;;
  esac
}

experiment_notes() {
  case "$1" in
    direct_ppo_pid_baseline_medium_seed0) echo "Fixed-task PID baseline, no dynamics or previous action." ;;
    direct_ppo_pid_dynprev_medium_seed0) echo "Fixed-task PID with dynamics and previous-action observations." ;;
    direct_ppo_pid_dynprev_net128_medium_seed0) echo "Fixed-task PID dynprev with pi/vf net_arch [128,128]." ;;
    direct_ppo_directrpm_dynprev_medium_seed0) echo "Experimental fixed-task direct-RPM dynprev comparison." ;;
    direct_ppo_pid_dynprev_taskdist_medium_medium_seed0) echo "PID dynprev with medium task distribution." ;;
    direct_ppo_pid_dynprev_net128_taskdist_medium_medium_seed0) echo "PID dynprev net128 with medium task distribution." ;;
    direct_ppo_directrpm_dynprev_taskdist_medium_medium_seed0) echo "Experimental direct-RPM dynprev with medium task distribution." ;;
    direct_ppo_directrpm_dynprev_net128_taskdist_medium_medium_seed0) echo "Experimental direct-RPM dynprev net128 with medium task distribution." ;;
    direct_ppo_pid_dynprev_taskdist_medium_low_lr_medium_seed0) echo "PID taskdist low learning-rate micro-HPO." ;;
    direct_ppo_pid_dynprev_taskdist_medium_ent005_medium_seed0) echo "PID taskdist entropy 0.005 micro-HPO." ;;
    direct_ppo_pid_dynprev_net128_taskdist_medium_low_lr_medium_seed0) echo "PID net128 taskdist low learning-rate micro-HPO." ;;
    direct_ppo_pid_dynprev_net128_taskdist_medium_ent005_medium_seed0) echo "PID net128 taskdist entropy 0.005 micro-HPO." ;;
    direct_ppo_directrpm_dynprev_taskdist_medium_low_lr_medium_seed0) echo "Experimental direct-RPM taskdist low learning-rate micro-HPO." ;;
    direct_ppo_directrpm_dynprev_taskdist_medium_ent005_medium_seed0) echo "Experimental direct-RPM taskdist entropy 0.005 micro-HPO." ;;
    curriculum_manual_pid_dynprev_taskdist_medium_medium_seed0) echo "Manual PID curriculum, 5 fixed-budget stages, medium task distribution base." ;;
    curriculum_manual_directrpm_dynprev_taskdist_medium_medium_seed0) echo "Experimental manual direct-RPM curriculum, 5 fixed-budget stages." ;;
    curriculum_llm_local_pid_dynprev_taskdist_medium_medium_seed0) echo "Local LLM PID curriculum, 10 adaptive budget stages, medium task distribution references allowed." ;;
    curriculum_llm_local_directrpm_dynprev_taskdist_medium_medium_seed0) echo "Optional experimental local LLM direct-RPM curriculum, 10 conservative adaptive budget stages." ;;
    *) return 1 ;;
  esac
}

lane_experiments() {
  case "$1" in
    1) echo "curriculum_llm_local_pid_dynprev_taskdist_medium_medium_seed0 direct_ppo_pid_baseline_medium_seed0" ;;
    2) echo "curriculum_manual_pid_dynprev_taskdist_medium_medium_seed0 direct_ppo_pid_dynprev_medium_seed0 direct_ppo_pid_dynprev_taskdist_medium_medium_seed0 direct_ppo_pid_dynprev_taskdist_medium_low_lr_medium_seed0 direct_ppo_pid_dynprev_taskdist_medium_ent005_medium_seed0 direct_ppo_pid_dynprev_net128_medium_seed0 direct_ppo_pid_dynprev_net128_taskdist_medium_medium_seed0" ;;
    3) echo "curriculum_llm_local_directrpm_dynprev_taskdist_medium_medium_seed0 direct_ppo_directrpm_dynprev_medium_seed0" ;;
    4) echo "direct_ppo_pid_dynprev_net128_taskdist_medium_low_lr_medium_seed0 direct_ppo_pid_dynprev_net128_taskdist_medium_ent005_medium_seed0 curriculum_manual_directrpm_dynprev_taskdist_medium_medium_seed0 direct_ppo_directrpm_dynprev_taskdist_medium_medium_seed0 direct_ppo_directrpm_dynprev_net128_taskdist_medium_medium_seed0 direct_ppo_directrpm_dynprev_taskdist_medium_low_lr_medium_seed0 direct_ppo_directrpm_dynprev_taskdist_medium_ent005_medium_seed0" ;;
    *) return 1 ;;
  esac
}
