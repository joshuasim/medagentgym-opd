"""Read-only audit of finished runs (Nov 14 methods check). Run on Torch from the repo folder:
    $SCRATCH/capstone/trainenv/bin/python scripts/audit_runs.py
Prints:
  B1  the key settings each RL run actually used (from VeRL's config dump in its train log)
  B4  per-step GRPO group signal: groups with 0/8, 8/8 or mixed results (only mixed groups give GRPO a gradient),
      and unsolved episodes that ended before 15 turns (time or token budget; estimate from the dumped text)
  B6  every eval summary in results/ (all / valid / clean, both prompt formats)
"""
import collections, glob, json, os, re

S = os.environ.get("SCRATCH", "")
RUNS = ["grpo_2gpu_s0", "opd_coder32b_v2_s0", "opd_mix_coder32b_s0"]
KEYS = ["total_epochs", "train_batch_size", "ppo_mini_batch_size", "temperature", "use_kl_loss", "kl_loss_coef",
        "use_task_rewards", "loss_mode", "loss_max_clamp", "lora_rank", "model_path", "default_local_dir"]

print("== B1: settings from each run's config dump")
for r in RUNS:
    logs = sorted(glob.glob(f"{S}/capstone/runs/{r}/train_*.log"))
    if not logs:
        print(f"{r}: no log yet"); continue
    txt = open(logs[0], errors="ignore").read(3_000_000)
    found = {}
    for k in KEYS:
        m = re.search(rf"'{k}': ([^,\n}}]+)", txt)
        if m: found[k] = m.group(1).strip()
    n = re.search(r"'rollout':.*?'n': (\d+)", txt, re.S)
    if n: found["rollout.n"] = n.group(1)
    print(f"{r} ({len(logs)} job log(s)): {found}")

print("\n== B4: GRPO group signal per step (groups of 8 rollouts per prompt)")
for r in ["grpo_2gpu_s0", "opd_coder32b_v2_s0", "opd_mix_coder32b_s0"]:
    files = sorted(glob.glob(f"{S}/capstone/runs/{r}/rollouts/*.jsonl"), key=lambda p: int(os.path.basename(p)[:-6]))
    if not files: continue
    print(f"-- {r}")
    tot = collections.Counter()
    for p in files:
        rows = [json.loads(l) for l in open(p) if l.strip()]
        groups = collections.defaultdict(list)
        early_time = early_tok = 0
        for x in rows:
            groups[str(x.get("uid", "")).rsplit("_", 2)[0]].append(x["score"] > 0)
            turns = x["output"].count("\nuser\n") + 1
            if x["score"] <= 0 and turns < 15:
                if len(x["output"]) > 25000: early_tok += 1
                else: early_time += 1
        c = collections.Counter("all_fail" if not any(g) else "all_pass" if all(g) else "mixed" for g in groups.values())
        tot.update(c)
        print(f"step {os.path.basename(p)[:-6]:>3}: groups {len(groups):>3} | all-fail {c['all_fail']:>3} | all-pass {c['all_pass']:>3} | "
              f"mixed {c['mixed']:>3} | unsolved & stopped early: ~time budget {early_time}, ~token budget {early_tok}")
    n = sum(tot.values())
    print(f"total: {n} groups | all-fail {tot['all_fail']} ({tot['all_fail']/n:.0%}) | all-pass {tot['all_pass']} "
          f"({tot['all_pass']/n:.0%}) | mixed {tot['mixed']} ({tot['mixed']/n:.0%})")

print("\n== B6: eval summaries (solved: all / valid / clean)")
for d in sorted(glob.glob("results/*/")):
    for fmt in ["merged_prompt", "system_prompt"]:
        f = os.path.join(d, fmt, "summary.json")
        if not os.path.exists(f): continue
        s = json.load(open(f))
        row = " / ".join(str(s.get(k, {}).get("solved", "-")) for k in ["all", "valid", "valid_clean"])
        print(f"{os.path.basename(d.rstrip('/')):<32} {fmt:<14} {row:<14} debugger={s.get('debugger')} mode={s.get('mode','test')}")
