"""Debug-action use during RL training, from VeRL's saved rollouts. Read-only; prints counts only.

runs/<run>/rollouts/<step>.jsonl has one line per training episode: input (prompt), output (the whole
multi-turn response as text: the model's turns and the environment's replies), score (reward), step, uid.
For each step it reports: episodes, solve rate (score >= 1, so a format bonus never counts as a solve),
share of episodes with >= 1 debug action, debug actions per episode, and the solve rate of episodes
with vs. without debugging. The last column compares them inside the same prompt group (same task,
8 rollouts), which controls for task difficulty: mean over groups that have both kinds of episodes.

Debug actions are counted from the JSON action the model writes ("action": "debug"). Heuristic: if a
debugger explanation itself quoted such JSON it would be counted too (rare).

Usage: python scripts/rollout_debug.py grpo_multi_lr3e5_s0 grpo_multi_lr3e5_s1 opd_multi_coder32b_lr3e5_s0
"""
import collections, glob, json, os, re, statistics as st, sys

DBG = re.compile(r'"action"\s*:\s*"debug"')


def load(run):
    d = os.path.join(os.environ.get("SCRATCH", "."), "capstone", "runs", run, "rollouts")
    rows = []
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            rows.append((int(r["step"]), hash(r["input"]), DBG.findall(r["output"] or "").__len__(), float(r["score"]) >= 1))
    return rows


def summarize(rows):
    n = len(rows)
    solved = sum(s for *_, s in rows)
    dbg = [r for r in rows if r[2] > 0]
    nodbg = [r for r in rows if r[2] == 0]
    rate = lambda xs: sum(x[3] for x in xs) / len(xs) if xs else float("nan")
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r[1]].append(r)
    diffs = [rate([r for r in g if r[2] > 0]) - rate([r for r in g if r[2] == 0])
             for g in groups.values() if any(r[2] > 0 for r in g) and any(r[2] == 0 for r in g)]
    return (n, solved / n, len(dbg) / n, sum(r[2] for r in rows) / n, rate(dbg), rate(nodbg),
            st.mean(diffs) if diffs else float("nan"), len(diffs))


def main():
    hdr = f"{'step':>5} {'eps':>5} {'solve':>6} {'dbg_eps':>8} {'dbg/ep':>7} {'solve|dbg':>10} {'solve|no':>9} {'in-group diff (n)':>18}"
    for run in sys.argv[1:]:
        rows = load(run)
        if not rows:
            print(f"== {run}: no rollouts found"); continue
        print(f"\n== {run}\n{hdr}")
        by = collections.defaultdict(list)
        for r in rows:
            by[r[0]].append(r)
        for step in sorted(by):
            n, s, de, dp, sd, sn, gd, gn = summarize(by[step])
            print(f"{step:5d} {n:5d} {s:6.3f} {de:8.3f} {dp:7.2f} {sd:10.3f} {sn:9.3f} {gd:+12.3f} ({gn:3d})")
        n, s, de, dp, sd, sn, gd, gn = summarize(rows)
        print(f"{'all':>5} {n:5d} {s:6.3f} {de:8.3f} {dp:7.2f} {sd:10.3f} {sn:9.3f} {gd:+12.3f} ({gn:3d})")


if __name__ == "__main__":
    main()
