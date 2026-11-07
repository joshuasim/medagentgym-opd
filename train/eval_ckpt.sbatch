#!/bin/bash
#SBATCH --job-name=eval-ckpt
#SBATCH --account=torch_pr_69_tandon_priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Merge an RL checkpoint (LoRA on top of the SFT model) into a full model, then run the standard
# BioCoder eval (157 / 78 / 30, both prompt formats) - the same eval as every baseline.
#   RUN_NAME=opd_coder32b_s0 sbatch train/eval_ckpt.sbatch                 (latest checkpoint)
#   RUN_NAME=opd_coder32b_s0 STEP=20 sbatch train/eval_ckpt.sbatch         (a specific step)
#   CKPT=<any global_step_N folder> TAG=... sbatch train/eval_ckpt.sbatch  (e.g. the smoke run's)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source torch/env.sh
# BASE_MODEL, not BASE: torch/env.sh uses $BASE for $SCRATCH/capstone and would overwrite it
BASE_MODEL="${BASE_MODEL:-$SCRATCH/capstone/models/qwen7b-sft-full}"   # the model the RL run started from
if [ -z "${CKPT:-}" ]; then
  CKPT="$SCRATCH/capstone/checkpoints/${RUN_NAME:?set RUN_NAME or CKPT}"
  [ -n "${STEP:-}" ] && CKPT="$CKPT/global_step_$STEP"
fi
TAG="${TAG:-${RUN_NAME:-ckpt}${STEP:+_step$STEP}}"
MERGED="$SCRATCH/capstone/models/rl_$TAG"
echo "checkpoint=$CKPT base=$BASE_MODEL -> $MERGED"
"$TRAIN_PY" "$REPO/scripts/merge_lora.py" --ckpt_dir "$CKPT" --base "$BASE_MODEL" --out "$MERGED"
[ -n "${SKIP_EVAL:-}" ] || MODEL="$MERGED" TAG="$TAG" bash torch/eval_baseline.sbatch
