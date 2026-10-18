# medagentgym-opd

Capstone code: On-Policy Distillation vs. GRPO for multi-turn BioCoder agents in MedAgentGym.

MedAgentGym (environment + eval) lives in its own clone: https://github.com/wshi83/MedAgentGym

## Layout

```
scripts/convert_sft.py         SampledTrajs (BioCoder) -> VeRL multi-turn SFT parquet
scripts/check_tasks.py         run every task's reference code twice: ok / error / empty / nondet
scripts/check_free_credit.py   does a dummy (wrong) function still get full credit?
data/sft/                      generated SFT data + stats + test overlap list
data/checks/                   per-task check results + valid task lists
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
  - Root cause (from MedAgentGym-Data `biocoder/train_data/` scripts): the 981 training tasks were generated from
    BioCoder's `python_hidden.json` (1,026 functions -> LLM-written question + context; 45 failed to parse -> 981).
    60 of the 157 test reference solutions are also in that hidden set, and no function-level dedup was done.
    The paper's 10-gram decontamination compares text, and the training questions are reworded, so it can't catch this.
  - `sft_full_*`  : all 351 (334 train / 17 val) — use for faithful reproduction
  - `sft_clean_*` : drops the 37 (299 / 15) — use for contamination-free runs
  - `test_overlap.json` : overlapping test `idx` values -> report every result on full (157) and clean (88) test sets

## Task validity checks (Oct 18, 2026)

How scoring works: `BiocoderTask.setup` runs the task's reference `code` and stores its printed
output (stdout + stderr, tracebacks included) as the answer. When the agent calls `validate_code`,
`base.py` inserts the agent's function into the task `context` and compares the printed output
to that answer by exact string match.

Env used: Python 3.11, MedAgentGym `requirements.txt` (numpy 2.2.4) + pysam, centrosome, torch.
Confirmed inside Docker (`ehr_gym:latest` + pysam/centrosome, see `docker/Dockerfile.capstone`): test 87 ok / 55 error / 15 nondet.
The one difference, `_reference_copies_pure`, prints a random number and happened to match across two runs, so it stays nondet.
Note: stock `ehr_gym:latest` lacks pysam and centrosome -> 74 test tasks fail on import (only 35 ok).

| | test (157) | train (981) |
|---|---|---|
| ok (runs, prints, deterministic) | 86 | 441 |
| error (reference code crashes) | 55 | 438 |
| empty (prints nothing) | 0 | 52 |
| nondet (output changes between runs) | 16 | 50 |
| dummy function gets full credit | 12 | 185 |
| **valid** (ok and not free credit) | **78** | **429** |
| valid and not overlapping train | 30 | — |

- Runtime-error answers include a random temp file path (`./cache/validation_code_<time>_<uuid>.py`), so they can never be matched.
- Compile-error contexts and empty-output contexts match any submission -> free credit.
- Under numpy 1.26 instead of 2.2.4, test `ok` rises 86 -> 101 (`np.asfarray` etc. removed in numpy 2): ground truth depends on the environment.
- 147 of the 351 SFT "successful" trajectories are on free-credit tasks, so their success doesn't show the code is correct.
- An earlier check (`check_biocoder.py`, using `exec`) reported 152/157 OK, but `exec` skips `if __name__ == "__main__": main()`, so it only checked that the code loads.

Lists: `data/checks/test_valid_subsets.json` (78 valid / 30 valid+clean), `data/checks/train_valid.json` (429 RL prompts).
