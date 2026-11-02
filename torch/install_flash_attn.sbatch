#!/bin/bash
#SBATCH --job-name=install-flash-attn
#SBATCH --account=torch_pr_69_tandon_priority   # fallback: torch_pr_69_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# One-time: install flash-attn into trainenv. VeRL's training path (SFT and the RL actor) imports it
# directly; verl[vllm] does not install it. pip first looks for a prebuilt wheel matching trainenv's
# torch/CUDA/Python; if none exists it compiles from source, which needs nvcc and takes 1-3 hours.
# We only build for A100 (sm_80) and H100/H200 (sm_90) to keep the compile short.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source torch/env.sh
BIN="$(dirname "$TRAIN_PY")"

nvidia-smi --query-gpu=name,driver_version --format=csv
"$TRAIN_PY" -c "import torch, sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda)"

# nvcc for a source build: try the cluster module first (version should match torch.version.cuda above)
if ! command -v nvcc >/dev/null 2>&1; then
  module avail cuda 2>&1 | tail -20
  module load cuda 2>/dev/null || true
fi
command -v nvcc && nvcc --version | tail -1 || echo "WARNING: no nvcc - a source build will fail; paste this log"

export MAX_JOBS=12 FLASH_ATTN_CUDA_ARCHS="80;90" NVCC_THREADS=2
export TMPDIR="$SCRATCH/capstone/tmp"; mkdir -p "$TMPDIR"
uv pip install -p "$TRAIN_PY" psutil packaging ninja einops wheel setuptools
time uv pip install -p "$TRAIN_PY" --no-build-isolation -v "flash-attn==2.8.3" 2>&1 | tail -40

"$TRAIN_PY" - <<'PY'
import torch, flash_attn
from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import unpad_input
q = torch.randn(10, 4, 64, device="cuda", dtype=torch.bfloat16)
cu = torch.tensor([0, 4, 10], device="cuda", dtype=torch.int32)
out = flash_attn_varlen_func(q, q, q, cu, cu, 6, 6, causal=True)
print("flash_attn", flash_attn.__version__, "OK on", torch.cuda.get_device_name(), tuple(out.shape))
PY
