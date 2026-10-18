"""Check whether BioCoder tasks produce a usable ground truth.

MedAgentGym scores BioCoder by running each task's reference `code` and taking its
output as the expected answer (BiocoderTask.setup -> validate_code). A task is only a
fair test / reward if that output is a real, repeatable answer. This script mirrors
validate_code (compile, append `print(answer)` rule, run as a subprocess, stdout+stderr)
and runs every task twice.

Status per task:
  ok        runs, prints something, same output both runs
  error     compile error or non-zero exit -> "answer" is an error message
  timeout   exceeds the time limit (env uses 120s)
  empty     runs but prints nothing -> any code that prints nothing would "match"
  nondet    output differs between two runs -> correct solutions can be scored wrong
Extra flags: mem_addr (output contains 0x... object addresses), stderr (warnings
mixed into the expected answer).

Usage:
  python scripts/check_tasks.py --tasks path/to/train_tasks.jsonl --out data/checks/train_check.json
"""

import argparse
import collections
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import tempfile

MEM_RE = re.compile(r"0x[0-9a-fA-F]{6,}")


def run_once(code, timeout, python):
    try:
        compile(code, "<string>", "exec")
    except Exception as e:  # mirrors validate_code's compile check
        return {"rc": "compile_error", "out": f"{type(e).__name__}: {e}", "stderr": ""}
    if "print" not in code and "answer" in code:  # same rule as validate_code
        code += "\nprint(answer)"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "task.py")
        with open(path, "w") as f:
            f.write(code)
        try:
            p = subprocess.run([python, path], cwd=d, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"rc": "timeout", "out": "", "stderr": ""}
    return {"rc": p.returncode, "out": p.stdout + ("\n" + p.stderr if p.stderr else ""), "stderr": p.stderr}


def check(task, timeout, python):
    r1 = run_once(task["code"], timeout, python)
    flags = []
    if r1["rc"] == "timeout":
        status = "timeout"
    elif r1["rc"] != 0:
        status = "error"
    elif not r1["out"].strip():
        status = "empty"
    else:
        r2 = run_once(task["code"], timeout, python)
        status = "ok" if r2["out"] == r1["out"] else "nondet"
        if MEM_RE.search(r1["out"]):
            flags.append("mem_addr")
        if r1["stderr"].strip():
            flags.append("stderr")
    sig = re.search(r"def\s+(\w+)", task.get("signature", ""))
    return {
        "idx": task["idx"],
        "function": sig.group(1) if sig else None,
        "status": status,
        "flags": flags,
        "output_head": r1["out"][:300],
        "output_tail": r1["out"][-300:],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(args.tasks) if l.strip()]
    results = []
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(check, t, args.timeout, args.python) for t in tasks]
        for i, f in enumerate(cf.as_completed(futs), 1):
            results.append(f.result())
            if i % 100 == 0:
                print(f"{i}/{len(tasks)}", file=sys.stderr)

    order = {t["idx"]: i for i, t in enumerate(tasks)}
    results.sort(key=lambda r: order[r["idx"]])
    summary = {
        "n": len(results),
        "status": dict(collections.Counter(r["status"] for r in results)),
        "flags": dict(collections.Counter(f for r in results for f in r["flags"])),
        "usable_ok": sum(r["status"] == "ok" and "mem_addr" not in r["flags"] for r in results),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "tasks": results}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
