# medagentgym-opd

Capstone code: On-Policy Distillation vs. GRPO for multi-turn BioCoder agents in MedAgentGym.

MedAgentGym (environment + eval) lives in its own clone: https://github.com/wshi83/MedAgentGym

## Layout

```
scripts/convert_sft.py   SampledTrajs (BioCoder) -> VeRL multi-turn SFT parquet
data/sft/                generated SFT data + stats + test overlap list
```

## SFT data

Regenerate:

```
hf download MedAgentGym/SampledTrajs --repo-type dataset --include "biocoder/*" --local-dir ../SampledTrajs
python scripts/convert_sft.py \
  --traj_dir ../SampledTrajs/biocoder \
  --test_tasks ../MedAgentGym/data/biocoder/test_tasks.jsonl \
  --train_tasks ../MedAgentGym/data/biocoder/train_tasks.jsonl \
  --out_dir data/sft
```

Format: one `messages` column (list of `{role, content}`), read by
`verl.utils.dataset.multiturn_sft_dataset`. Loss is applied to assistant turns only;
system prompt, task, and environment feedback (role `user`) are context.

### Findings (Oct 18, 2026)

- 351 BioCoder trajectories, all successful (score 1), all from training-set functions.
- Mostly single-turn: 76.6% solve on the first assistant turn; mean 1.75 assistant turns, max 12.
- `debug` action results are not recorded: a debug turn is followed directly by the next assistant turn.
- Length with Qwen2.5 chat template: median 709 tokens, p95 2,946, max 33,442 (2 trajectories > 8,192).
- **Train/test overlap in MedAgentGym's BioCoder split:** 69 of 157 test tasks share a function
  signature with a training task (60 of those 69 have an identical reference solution; problem text is reworded).
  37 SFT trajectories are on those functions.
  - `sft_full_*`  : all 351 (334 train / 17 val) — use for faithful reproduction
  - `sft_clean_*` : drops the 37 (299 / 15) — use for contamination-free runs
  - `test_overlap.json` : overlapping test `idx` values -> report every result on full (157) and clean (88) test sets
