"""Label the excluded-pair check runs (see make_excluded_check.py) exactly as
analyze.py labels main fault runs: against ALL three clean runs of the same
student. Prints one line per run and writes reports/phase1/excluded_check.csv.

    python experiments/analyze_excluded_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from analyze import (classify, load_trace, plan_signature, trace_harm,  # noqa: E402
                     world_signature)

METHODS = ["full_restart", "full_restart_oracle", "verify_all", "verify_all_self"]


def main():
    res_dir = ROOT / "results" / "excluded_check"
    tasks = {t["task_id"]: t for t in json.loads((ROOT / "data" / "tasks_excluded_check.json").read_text())["tasks"]}
    rows = [json.loads(l) for l in open(res_dir / "faults.jsonl")]
    out = []
    for r in sorted(rows, key=lambda r: r["task_id"]):
        task = tasks[r["task_id"]]
        base = task["base_task_id"]
        ftrace = load_trace(res_dir / "traces", r["run_id"])
        ctraces = [load_trace(ROOT / "results" / "traces", f"clean-{base}-r{i}") for i in range(3)]
        ctraces = [c for c in ctraces if c]
        harm = trace_harm(ftrace, len(task["expected"].get("drops", [])))
        if harm is not None:
            r["fault_harmful_irreversible"] = harm
        level = classify(r, ftrace, ctraces)
        planner_tok = max((c["completion_tokens"] for c in ftrace["llm_calls"] if c["stage"] == "planner"), default=0)
        o = {"task": r["task_id"], "base": base, "patch": task["faults"][r["fault_id"]]["description"],
             "level": level, "success": r["success"], "failures": "; ".join(r["check_failures"]),
             "plan_matches_a_clean_run": plan_signature(ftrace) in {plan_signature(c) for c in ctraces},
             "world_matches_a_clean_run": world_signature(ftrace) in {world_signature(c) for c in ctraces},
             "unauthorized": r["unauthorized_actions"], "harmful_irreversible": r["fault_harmful_irreversible"],
             "parse_failures": ",".join(r["parse_failures"]), "planner_completion_tok": planner_tok,
             "n_clean": len(ctraces)}
        o.update({f"{m}.success": r.get(f"{m}.success") for m in METHODS})
        out.append(o)
    df = pd.DataFrame(out)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60)
    print(df.drop(columns=["failures"]).to_string(index=False))
    for _, x in df[df.failures != ""].iterrows():
        print(f"  {x.task}: {x.failures}")
    df.to_csv(ROOT / "reports" / "phase1" / "excluded_check.csv", index=False)


if __name__ == "__main__":
    main()
