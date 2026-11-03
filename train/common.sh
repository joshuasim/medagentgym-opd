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
# Agent shell commands and task code write into the working directory, so use a throwaway folder
# (not the repo or MedAgentGym). opd/agent_loop.py chdirs Ray workers here too; validate_code writes ./cache/.
export MEDAGENTGYM_WORKDIR="$SCRATCH/capstone/work/${SLURM_JOB_ID:-local}_train"
mkdir -p "$MEDAGENTGYM_WORKDIR/cache" && cd "$MEDAGENTGYM_WORKDIR"
# torch/env.sh puts taskenv's `python` first on PATH (for executing task code), so the trainer
# must be launched with trainenv's interpreter explicitly.
PY="${TRAIN_PY:?source torch/env.sh first}"

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
  # Merge the LoRA into the base weights before each sync to vLLM, instead of loading it as a vLLM
  # adapter. The adapter path crashed in the Nov 3 smoke run: vLLM received an empty adapter
  # (StopIteration in _create_merged_loras_inplace). Merged sync uses the plain full-weight path.
  actor_rollout_ref.model.lora.merge=true
  actor_rollout_ref.actor.optim.lr=1e-5  # paper: 1e-5
  actor_rollout_ref.actor.ppo_mini_batch_size=32
  actor_rollout_ref.actor.use_dynamic_bsz=true          # pack by tokens; one episode is up to 2048+8192
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.temperature=0.6   # paper: 0.6
  actor_rollout_ref.rollout.n=8
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5
  # Weights are synced to vLLM in fp32. Qwen2.5-7B's embedding/lm_head are 2.18 GB each, larger than the
  # default 2048 MB bucket, so they take verl's "direct send" path, which only works for CUDA tensors and
  # crashes (IndexError in rebuild_ipc) when FSDP param offload keeps them on CPU. A 3 GB bucket avoids it.
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072
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
