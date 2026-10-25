#!/bin/bash
#SBATCH --job-name=biocoder-eval
#SBATCH --account=torch_pr_69_general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# Baseline eval on the 157 BioCoder test tasks. Submit from the repo folder:
#   sbatch torch/eval_baseline.sbatch                                   (base Qwen2.5-7B-Instruct)
#   MODEL=MedAgentGym/MedCopilot-7B TAG=medcopilot7b sbatch torch/eval_baseline.sbatch
# Runs twice: MedAgentGym's vLLM prompt format (system merged into user, like the paper's
# open-model runs) and the separate-system format our SFT/RL training uses.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source torch/env.sh
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TAG="${TAG:-base_qwen7b}"
OUT="$REPO/results/$TAG"
mkdir -p "$OUT"
cd "$MEDAGENTGYM_ROOT"   # validate_code writes ./cache here

nvidia-smi --query-gpu=name,memory.total --format=csv
"$(dirname "$TRAIN_PY")/vllm" serve "$MODEL" --port 8000 --max-model-len 32768 \
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
