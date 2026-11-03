#!/bin/bash
#SBATCH --job-name=biocoder-eval
#SBATCH --account=torch_pr_69_tandon_priority   # approved by Prof. Tan (Oct 27); fallback: torch_pr_69_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Baseline eval on the 157 BioCoder test tasks. Submit from the repo folder:
#   sbatch torch/eval_baseline.sbatch                                   (base Qwen2.5-7B-Instruct)
#   MODEL=MedAgentGym/MedCopilot-7B TAG=medcopilot7b sbatch torch/eval_baseline.sbatch
#   MODEL=Qwen/Qwen2.5-32B-Instruct TAG=base_qwen32b TP=2 sbatch --gres=gpu:2 --time=04:00:00 torch/eval_baseline.sbatch
# Runs twice: MedAgentGym's vLLM prompt format (system merged into user, like the paper's
# open-model runs) and the separate-system format our SFT/RL training uses.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source torch/env.sh
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TAG="${TAG:-base_qwen7b}"
TP="${TP:-1}"   # tensor parallel GPUs for vLLM; 32B needs TP=2 plus: sbatch --gres=gpu:2
OUT="$REPO/results/$TAG"
mkdir -p "$OUT"
# Run from a throwaway folder: the agent's terminal commands and the task code write files into the
# working directory (on Nov 2 they left venvs, CNVkit source and csv files in MedAgentGym/). An empty
# folder also matches how scripts/check_tasks.py audited the tasks. validate_code writes ./cache here.
WORK="$SCRATCH/capstone/work/${SLURM_JOB_ID:-local}_$TAG"
mkdir -p "$WORK/cache" && cd "$WORK"

nvidia-smi --query-gpu=name,memory.total --format=csv
"$(dirname "$TRAIN_PY")/vllm" serve "$MODEL" --port 8000 --max-model-len 32768 --tensor-parallel-size "$TP" \
  --gpu-memory-utilization 0.9 --dtype bfloat16 --served-model-name "$MODEL" \
  > "$OUT/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT
for i in $(seq 1 120); do
  curl -sf localhost:8000/health >/dev/null && break
  sleep 10
done
curl -sf localhost:8000/health >/dev/null || { echo "vLLM failed to start"; tail -50 "$OUT/vllm.log"; exit 1; }
echo "vLLM up after $((i*10))s"

"$TRAIN_PY" "$REPO/scripts/eval_biocoder.py" --model "$MODEL" --out "$OUT/merged_prompt" --merge_system --save_history
"$TRAIN_PY" "$REPO/scripts/eval_biocoder.py" --model "$MODEL" --out "$OUT/system_prompt" --save_history
echo "=== merged prompt (paper format) ===";  cat "$OUT/merged_prompt/summary.json"
echo "=== system prompt (training format) ==="; cat "$OUT/system_prompt/summary.json"
