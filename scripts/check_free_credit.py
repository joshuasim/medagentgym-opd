"""Find BioCoder tasks where a WRONG answer still gets full credit.

The env inserts the agent's function into the task context (base.py: context.replace(
"<<insert solution here>>", code)) and compares the printed output with the reference
output. If the context has a syntax error before the insertion point, or prints nothing,
or never depends on the function, then ANY function matches. This script inserts a dummy
`def f(*args, **kwargs): return None` and reports tasks where it matches the reference.

Usage (after check_tasks.py):
  python scripts/check_free_credit.py TASKS.jsonl data/checks/X_check.json data/checks/X_free_credit.json
"""
import collections
import concurrent.futures as cf
import json
import re
import sys

sys.path.insert(0, "scripts")
from check_tasks import run_once  # noqa: E402

tasks = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
chk = {t["idx"]: t for t in json.load(open(sys.argv[2]))["tasks"]}


def go(t):
    status = chk[t["idx"]]["status"]
    if status not in ("error", "empty", "ok"):
        return t["idx"], status, False
    name = re.search(r"def\s+(\w+)", t["signature"]).group(1)
    dummy = f"def {name}(*args, **kwargs):\n    return None"
    ref = run_once(t["code"], 120, sys.executable)["out"]
    got = run_once(t["context"].replace("<<insert solution here>>", "\n" + dummy + "\n"), 120, sys.executable)["out"]
    return t["idx"], status, ref == got


with cf.ThreadPoolExecutor(16) as ex:
    res = list(ex.map(go, tasks))
print(collections.Counter((s, ok) for _, s, ok in res))
json.dump([r for r in res if r[2]], open(sys.argv[3], "w"))
