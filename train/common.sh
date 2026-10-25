# Shared settings for GRPO and OPD runs. DRAFT: config keys checked against verl 0.9.0's
# generated trainer config, but not yet run on GPUs. Expect to tune batch sizes / memory.
#
# Required env:
#   MEDAGENTGYM_ROOT  path to the MedAgentGym clone (task files + ehr_gym package)
#   STUDENT_PATH      SFT checkpoint to start from (both GRPO and OPD start here)
#   N_GPUS            GPUs on the node
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$REPO:${MEDAGENTGYM_ROOT}:${PYTHONPATH:-}"
export PYTHONHASHSEED=0   # set iteration order is part of some task outputs; fix it so answers are stable
cd "$REPO"                # validate_code writes ./cache/

COMMON_ARGS=(
  data.train_files="$REPO/data/rl/train_valid.parquet"
  data.val_files="$REPO/data/rl/test_all.parquet"
  data.max_prompt_length=2048
  data.max_response_length=8192          # whole multi-turn episode (assistant + observations)
  data.train_batch_size=32
  data.return_raw_chat=true
  actor_rollout_ref.model.path="$STUDENT_PATH"
  actor_rollout_ref.model.lora_rank=64
  actor_rollout_ref.model.lora_alpha=128
  actor_rollout_ref.actor.optim.lr=1e-5  # paper: 1e-5
  actor_rollout_ref.actor.ppo_mini_batch_size=32
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.temperature=0.6   # paper: 0.6
  actor_rollout_ref.rollout.n=8
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.multi_turn.enable=true
  actor_rollout_ref.rollout.agent.default_agent_loop=medagentgym
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$REPO/configs/agent_loop.yaml"
  trainer.n_gpus_per_node="${N_GPUS}"
  trainer.nnodes=1
  trainer.logger='["console"]'
  trainer.test_freq=10
  trainer.save_freq=10
  trainer.total_epochs=5
)
