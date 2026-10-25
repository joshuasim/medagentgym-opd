# Source this in every Torch job:  source torch/env.sh
BASE="$SCRATCH/capstone"
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MEDAGENTGYM_ROOT="$BASE/MedAgentGym"
export HF_HOME="$BASE/hf_cache"
export UV_CACHE_DIR="$BASE/uv_cache"
export TRAIN_PY="$BASE/trainenv/bin/python"
# task code is executed as `python <file>` -> taskbin/python runs taskenv (MedAgentGym's packages).
# trainenv/bin comes next so vLLM/torch.compile can find tools like ninja.
export PATH="$BASE/taskbin:$BASE/trainenv/bin:$HOME/.local/bin:$PATH"
export PYTHONPATH="$REPO:$MEDAGENTGYM_ROOT:${PYTHONPATH:-}"
export PYTHONHASHSEED=0   # some task outputs print sets; fix iteration order so answers are stable
export TOKENIZERS_PARALLELISM=false
