#!/bin/bash
# GRPO reproduction (correctness reward from the MedAgentGym env). KL reference = the SFT model itself
# (VeRL uses the actor with the LoRA adapter switched off, so no extra model in memory).
source "$(dirname "$0")/common.sh"
"$PY" -m verl.trainer.main_ppo "${COMMON_ARGS[@]}" \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=1e-3 \
  trainer.project_name=medagentgym-opd trainer.experiment_name="${RUN_NAME}" \
  "$@"
