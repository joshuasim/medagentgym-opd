"""Replay the released SFT trajectories through opd.biocoder_env and check that
(a) the environment's observations match the user messages recorded in the trajectory, and
(b) the final answer is scored correct, on tasks our checks mark as valid.
This is evidence our env reproduces MedAgentGym's behaviour on real data."""
import concurrent.futures as cf, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opd.biocoder_env import BioCoderEpisode, load_tasks

ROOT = os.environ.get("MEDAGENTGYM_ROOT", "../MedAgentGym"); DATA = os.path.join(ROOT, "data", "biocoder")
TRAJ = sys.argv[1]
train = load_tasks(DATA, "train"); by_q = {t["problem"].strip(): i for i, t in enumerate(train)}
valid = set(json.load(open("data/checks/train_valid.json"))["valid_idx"])

def norm(t):
    """Strip run-specific details: temp file paths, uuids, timings, install locations."""
    t = re.sub(r'File "[^"]*"', 'File "<path>"', t)
    t = re.sub(r"validation_code_\w+\.py", "<file>", t)
    t = re.sub(r"/[\w./-]*site-packages/", "<site>/", t)
    t = re.sub(r"\d+\.\d+s", "<t>", t)
    return re.sub(r"\s+", " ", t).strip()

mism = []
def replay(f):
    d = json.load(open(f)); msgs = [m for m in d if "role" in m]
    i = by_q[msgs[1]["content"].strip()]
    if train[i]["idx"] not in valid: return None
    ep = BioCoderEpisode(DATA, "train", i); obs_match = obs_total = 0; last = None
    rest = msgs[2:]
    for k, m in enumerate(rest):
        if m["role"] != "assistant": continue
        r = ep.step(m["content"])
        if r is None:  # debug: debugger reply isn't recorded, skip
            r = ep.finish_debug(""); continue
        last = r
        nxt = rest[k + 1] if k + 1 < len(rest) else None
        if nxt is not None and nxt["role"] == "user":
            obs_total += 1; ok = norm(nxt["content"]) == norm(r.observation); obs_match += int(ok)
            if not ok: mism.append((os.path.basename(f), nxt["content"][-200:], r.observation[-200:]))
    return f, bool(last and last.done), obs_match, obs_total

files = sorted(glob.glob(os.path.join(TRAJ, "*.jsonl")))
with cf.ThreadPoolExecutor(8) as ex: res = [r for r in ex.map(replay, files) if r]
solved = sum(r[1] for r in res); om = sum(r[2] for r in res); ot = sum(r[3] for r in res)
print(f"trajectories on valid tasks: {len(res)}  final answer scored correct: {solved}")
print(f"intermediate observations identical to recorded ones: {om}/{ot}")
print("not solved:", [os.path.basename(r[0]) for r in res if not r[1]][:20])
print("normalized mismatches:", len(mism))
for m in mism[:3]: print("----", m[0]); print("RECORDED:", repr(m[1])); print("OURS:    ", repr(m[2]))
