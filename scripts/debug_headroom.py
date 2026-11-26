"""How much room is there for a better debugger? Read-only, counts only (BioCoder outputs are public).

For each eval result it splits the unsolved tasks by what the agent saw last:
  crash        - the last observation is a Python error/traceback (a debugger could help)
  wrong_output - the code ran and printed something, but the answer was wrong
                 (MedAgentGym never says "wrong", so a debugger is never consulted for these)
  other        - LLM error / no observation recorded
and counts debug usage: tasks with >=1 debug call, and tasks solved after a debug call.

Usage: python scripts/debug_headroom.py sft_multi_e2 grpo_multi_lr3e5_s0_step26 ... [--fmt system_prompt]
"""
import argparse, collections, json, os, re

ERR = re.compile(r"Traceback \(most recent call last\)|\b\w+Error\b|\bException\b|SyntaxError|timed out|Error: ", re.I)


def last_obs(rec):
    hist = rec.get("history") or []
    users = [m["content"] for m in hist if m["role"] == "user"]
    return users[-1] if len(users) > 1 else None   # users[0] is the task prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--fmt", default="system_prompt")
    a = ap.parse_args()
    print(f"{'model':38s} {'solved':>6s} {'unsolved':>8s} {'crash':>6s} {'wrong':>6s} {'other':>6s} | {'dbg_tasks':>9s} {'dbg_calls':>9s} {'solved_after_dbg':>16s}")
    for t in a.tags:
        p = os.path.join("results", t, a.fmt, "per_task.jsonl")
        if not os.path.exists(p):
            print(f"{t:38s} missing {p}"); continue
        rows = [json.loads(l) for l in open(p)]
        c = collections.Counter()
        for r in rows:
            acts = r.get("actions", [])
            nd = acts.count("debug")
            c["dbg_calls"] += nd
            c["dbg_tasks"] += nd > 0
            if r["solved"]:
                c["solved"] += 1
                if "debug" in acts:
                    c["solved_after_dbg"] += 1
                continue
            o = last_obs(r)
            if r["stop"].startswith("llm_error") or o is None:
                c["other"] += 1
            elif ERR.search(o):
                c["crash"] += 1
            else:
                c["wrong"] += 1
        uns = len(rows) - c["solved"]
        print(f"{t:38s} {c['solved']:6d} {uns:8d} {c['crash']:6d} {c['wrong']:6d} {c['other']:6d} | {c['dbg_tasks']:9d} {c['dbg_calls']:9d} {c['solved_after_dbg']:16d}")


if __name__ == "__main__":
    main()
