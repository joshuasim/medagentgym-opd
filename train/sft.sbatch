#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --account=torch_pr_69_tandon_priority   # fallback: torch_pr_69_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x-%j.out
# LoRA SFT warm-up of Qwen2.5-7B-Instruct on the BioCoder trajectories, then merge and evaluate.
# Submit from the repo folder:
#   sbatch train/sft.sbatch                      (full set: 334 train / 17 val, paper-faithful)
#   DATA=clean sbatch train/sft.sbatch           (drops the 37 trajectories on test-overlapping functions)
#   SKIP_EVAL=1 sbatch train/sft.sbatch          (train + merge only)
# Paper (MedAgentGym App. E): AdamW, lr 1e-4, batch size 8. Epochs are not stated; we use 3.
# Max length 8192: trajectories longer than that (2 of 334 in the full set) are dropped, not truncated,
# because truncation would cut off the final answer (scripts/filter_sft_length.py).
# Outputs:
#   $SCRATCH/capstone/checkpoints/sft_$DATA/   VeRL checkpoint (last step only)
#   $SCRATCH/capstone/models/qwen7b-sft-$DATA/ merged HF model  <- STUDENT_PATH for OPD / GRPO
#   results/sft_$DATA/                          eval on 157 / 78 / 30 (both prompt formats)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source torch/env.sh
# VeRL's trainer loads the model with flash_attention_2; setup.sh installs verl[vllm], which does not pull flash-attn
"$TRAIN_PY" -c "import flash_attn, peft" || { echo "flash_attn or peft missing in trainenv (see README: SFT)"; exit 1; }
DATA="${DATA:-full}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
EPOCHS="${EPOCHS:-3}"
CKPT="$SCRATCH/capstone/checkpoints/sft_$DATA"
MERGED="$SCRATCH/capstone/models/qwen7b-sft-$DATA"
BIN="$(dirname "$TRAIN_PY")"
mkdir -p "$SCRATCH/capstone/models" "$CKPT"

nvidia-smi --query-gpu=name,memory.total --format=csv
# fp32 master weights for 7B are ~30 GB; on a 40 GB card keep the frozen base on CPU instead
GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
OFFLOAD=false; [ "$GPU_MB" -lt 60000 ] && OFFLOAD=true
echo "GPU memory ${GPU_MB} MiB -> param_offload=$OFFLOAD"

SFT_DATA="$CKPT/data"
"$TRAIN_PY" "$REPO/scripts/filter_sft_length.py" --tokenizer "$BASE_MODEL" --max_tokens 8192 --out_dir "$SFT_DATA" \
  "$REPO/data/sft/sft_${DATA}_train.parquet" "$REPO/data/sft/sft_${DATA}_val.parquet"

echo "=== SFT ($DATA, $EPOCHS epochs) ==="
"$BIN/torchrun" --standalone --nnodes=1 --nproc_per_node=1 -m verl.trainer.sft_trainer \
  data.train_files="$SFT_DATA/sft_${DATA}_train.parquet" \
  data.val_files="$SFT_DATA/sft_${DATA}_val.parquet" \
  data.messages_key=messages \
  data.train_batch_size=8 \
  data.micro_batch_size_per_gpu=4 \
  data.use_dynamic_bsz=true \
  data.max_token_len_per_gpu=16384 \
  data.max_length=8192 \
  data.truncation=error \
  data.num_workers=4 \
  model.path="$BASE_MODEL" \
  model.lora_rank=64 \
  model.lora_alpha=128 \
  model.target_modules=all-linear \
  model.enable_gradient_checkpointing=true \
  engine.param_offload=$OFFLOAD \
  optim.lr=1e-4 \
  optim.weight_decay=0.0 \
  optim.lr_warmup_steps_ratio=0.05 \
  optim.lr_scheduler_type=cosine \
  trainer.total_epochs="$EPOCHS" \
  trainer.test_freq=10 \
  trainer.save_freq=-1 \
  trainer.resume_mode=disable \
  trainer.default_local_dir="$CKPT" \
  trainer.project_name=medagentgym-opd \
  trainer.experiment_name="sft_$DATA" \
  trainer.logger='["console"]' \
  trainer.n_gpus_per_node=1

echo "=== merge LoRA -> $MERGED ==="
"$TRAIN_PY" "$REPO/scripts/merge_lora.py" --ckpt_dir "$CKPT" --base "$BASE_MODEL" --out "$MERGED"

if [ -z "${SKIP_EVAL:-}" ]; then
  echo "=== eval ==="
  MODEL="$MERGED" TAG="sft_$DATA" bash torch/eval_baseline.sbatch
fi
