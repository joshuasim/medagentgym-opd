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
export TQDM_DISABLE=1   # MedAgentGym prints a progress bar per code run; silence it in logs
# Caches on scratch, not /home (Nov 19: /home hit its 30,000-file quota, mostly ~/.cache/uv, ~/.cache/vllm,
# ~/.cache/pip). XDG_CACHE_HOME covers pip, vLLM, flashinfer and torch; Triton has its own variable.
export XDG_CACHE_HOME="$BASE/cache"
export TRITON_CACHE_DIR="$BASE/cache/triton"
mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR"

# The agent's `terminal` action runs arbitrary shell commands with this PATH. On Nov 2 an agent ran
# `pip install ...` during an eval: `pip` resolved to trainenv's pip, which started replacing torch/numpy/
# scipy, and the job was killed mid-uninstall, deleting torch from trainenv. Block package managers for
# everything launched from a job: shims in taskbin (first on PATH) refuse, and PIP_NO_INDEX stops
# `python -m pip install` before it uninstalls anything. Our own installs use uv, which ignores PIP_*.
for _c in pip pip3 pip3.11 conda mamba easy_install; do
  if [ ! -x "$BASE/taskbin/$_c" ]; then
    printf '#!/bin/sh\necho "ERROR: installing packages is disabled in this environment." >&2\nexit 1\n' > "$BASE/taskbin/$_c"
    chmod +x "$BASE/taskbin/$_c"
  fi
done
unset _c
export PIP_NO_INDEX=1
