"""Scan all traces + result rows for LLM response problems.
Usage: python experiments/audit_llm.py [repo_root]

Traces keep token counts but not raw replies, so "hit limit" means
completion_tokens >= the role's max_tokens (configs/experiment.yaml)."""
import json, sys, glob, os, collections

root = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIMIT = {"researcher": 12000, "planner": 4096, "executor": 4096, "verifier": 16000}
NEAR = 0.95

def limit_for(stage, fname):
    if "obsolete" in fname and stage == "verifier":
        return 4096
    return LIMIT.get(stage, 4096)

calls = collections.defaultdict(list)        # (stage, verifier-kind) -> completion tokens
hits, near, tiny, pf = [], [], [], []
n_traces = 0
for f in sorted(glob.glob(f"{root}/results/traces/*.json") + glob.glob(f"{root}/results/obsolete/traces/*.json")):
    n_traces += 1
    t = json.load(open(f)); name = os.path.relpath(f, f"{root}/results")
    for c in t.get("llm_calls", []):
        st = c["stage"]; ct = c["completion_tokens"]; lim = limit_for(st, name)
        kind = st
        if st == "verifier":
            kind = "verifier-self" if "self" in name else ("verifier-strong-4096(obsolete)" if "obsolete" in name else "verifier-strong")
        calls[kind].append(ct)
        if ct >= lim: hits.append((name, st, ct, lim))
        elif ct >= NEAR * lim: near.append((name, st, ct, lim))
        if ct < 20: tiny.append((name, st, ct, c["prompt_tokens"]))
    for a in t.get("artifacts", []):
        if (a.get("meta") or {}).get("parse_failed"):
            pf.append((name, a["producer"], a.get("meta")))

print(f"traces scanned: {n_traces}\n")
print("completion tokens by role:  n  median  p95  max  (limit)")
for k, v in sorted(calls.items()):
    v = sorted(v); lim = 4096 if "obsolete" in k else LIMIT[k.split("-")[0]]
    print(f"  {k:32s} {len(v):5d} {v[len(v)//2]:6d} {v[int(.95*(len(v)-1))]:6d} {v[-1]:6d}  ({lim})")
for title, rows in [("HIT LIMIT (completion >= max_tokens)", hits), (f"NEAR LIMIT (>= {int(NEAR*100)}%)", near),
                    ("TINY/EMPTY replies (<20 completion tokens)", tiny), ("PARSE FAILED artifacts", pf)]:
    print(f"\n== {title}: {len(rows)}")
    for r in rows: print("  ", r)

# result rows: errors + verifier verdict parse failures
print("\n== result rows")
for f in sorted(glob.glob(f"{root}/results/*.jsonl") + glob.glob(f"{root}/results/obsolete/*.jsonl")):
    n = err = vpf = 0; errs = collections.Counter()
    for l in open(f):
        r = json.loads(l); n += 1
        for k, v in r.items():
            if (k == "error" or k.endswith(".error")) and v:
                err += 1; errs[str(v)[:120]] += 1
            if k.endswith("verdicts") or k == "verdicts":
                vpf += sum(1 for x in v or [] if x.get("parse_failed"))
        m = r.get("metrics") or {}
        for x in m.get("verdicts", []) or []:
            vpf += bool(x.get("parse_failed"))
    print(f"  {os.path.relpath(f, root)}: rows={n} errors={err} verifier_verdict_parse_failures={vpf}")
    for e, c in errs.most_common(): print(f"     {c}x {e}")
