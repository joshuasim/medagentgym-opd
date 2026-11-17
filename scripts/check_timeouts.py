"""Does the training-only 30 s code cap mark correct-but-slow solutions as failures?

Training rollouts cap each validate_code run at 30 s (configs/agent_loop.yaml code_timeout); evaluation and
MedAgentGym use 120 s. A correct solution takes roughly as long as the task's reference code, so:

  Part 1  times every RL training task's reference code (same way validate_code runs it: `python file.py`,
          120 s limit), with --workers runs in parallel because training also runs many episodes at once.
          Tasks whose reference needs more than 30 s can never be rewarded in training, even when solved.
  Part 2  scans finished runs' rollout dumps for MedAgentGym's timeout message with the 30 s limit and counts
          how many episodes / tasks hit it, and how many of those are tasks from Part 1's slow list.

Prints task idx and timings only (BioCoder is public, CC-BY-4.0). Run on Torch from the repo folder, on a CPU
allocation (not the login node):
  source torch/env.sh
  srun --account=torch_pr_69_tandon_priority --cpus-per-task=16 --mem=32G --time=01:00:00 \
      $TRAIN_PY scripts/check_timeouts.py --out data/checks/train_ref_times.json
"""
import argparse, collections, concurrent.futures as cf, glob, json, os, subprocess, tempfile, time

import pandas as pd

S = os.environ.get("SCRATCH", "")


def time_reference(task, limit):
    code = task["code"]
    if "print" not in code and "answer" in code:  # same rule as validate_code
        code += "\nprint(answer)"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ref.py")
        with open(path, "w") as f:
            f.write(code)
        t0 = time.monotonic()
        try:
            p = subprocess.run(["python", path], cwd=d, capture_output=True, text=True, timeout=limit)
            status = "ok" if p.returncode == 0 else "error"
        except subprocess.TimeoutExpired:
            status = "timeout"
        return {"idx": task["idx"], "seconds": round(time.monotonic() - t0, 1), "status": status}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--cap", type=float, default=30)
    ap.add_argument("--runs", nargs="*", default=["grpo_2gpu_s0", "opd_coder32b_v2_s0"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = os.environ["MEDAGENTGYM_ROOT"]
    tasks = [json.loads(l) for l in open(os.path.join(root, "data/biocoder/train_tasks.jsonl")) if l.strip()]
    rl = pd.read_parquet("data/rl/train_valid.parquet")
    rl_idx = [str(e["idx"]) for e in rl["extra_info"]]
    by_idx = {str(t["idx"]): t for t in tasks}
    todo = [by_idx[i] for i in rl_idx]

    print(f"== Part 1: reference run time, {len(todo)} RL training tasks, {args.workers} in parallel, limit {args.limit} s")
    res = []
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for r in ex.map(lambda t: time_reference(t, args.limit), todo):
            res.append(r)
    secs = sorted(r["seconds"] for r in res)
    q = lambda f: secs[min(len(secs) - 1, int(f * len(secs)))]
    print(f"seconds p50 {q(.5)} p90 {q(.9)} p99 {q(.99)} max {secs[-1]} | status {dict(collections.Counter(r['status'] for r in res))}")
    slow = sorted((r for r in res if r["seconds"] > args.cap * 2 / 3), key=lambda r: -r["seconds"])
    over = [r for r in slow if r["seconds"] > args.cap]
    print(f"over {args.cap:.0f} s (never rewardable in training): {len(over)} | "
          f"{args.cap * 2 / 3:.0f}-{args.cap:.0f} s (at risk when the node is busy): {len(slow) - len(over)}")
    for r in slow:
        print(f"  idx {r['idx']:<12} {r['seconds']:>6.1f} s  {r['status']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"cap": args.cap, "workers": args.workers, "tasks": res}, f, indent=1)
        print(f"written {args.out}")

    print(f"\n== Part 2: rollouts that hit the {args.cap:.0f} s cap")
    over_idx = {r["idx"] for r in over}
    msg = f"(timeout limit: {int(args.cap)}s)"  # MedAgentGym _run_code_file message for a capped run
    prompts = [(next(m["content"] for m in p if m["role"] == "user").strip()[:200], i) for p, i in zip(rl["prompt"], rl_idx)]
    for run in args.runs:
        files = glob.glob(f"{S}/capstone/runs/{run}/rollouts/*.jsonl")
        if not files:
            print(f"{run}: no rollout dumps"); continue
        n_ep = n_hit = n_hit_solved = 0
        hit_tasks = collections.Counter()
        for fpath in files:
            for line in open(fpath):
                if not line.strip():
                    continue
                x = json.loads(line)
                n_ep += 1
                if msg in x["output"]:
                    n_hit += 1
                    n_hit_solved += x["score"] >= 1
                    idx = next((i for k, i in prompts if k in x["input"]), "?")
                    hit_tasks[idx] += 1
        in_over = sum(c for i, c in hit_tasks.items() if i in over_idx)
        print(f"{run}: {n_hit}/{n_ep} episodes hit the cap ({n_hit_solved} of them still solved) | "
              f"{len(hit_tasks)} tasks | {in_over} of these episodes are on tasks whose reference takes > {args.cap:.0f} s")
        print("  most often:", ", ".join(f"{i} ({c})" for i, c in hit_tasks.most_common(8)))


if __name__ == "__main__":
    main()
