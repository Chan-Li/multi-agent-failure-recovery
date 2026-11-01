"""Phase 1: run each fault schedule, measure propagation, then apply the
baseline recovery methods (full restart, verify-all) under the same schedule."""
from __future__ import annotations

import argparse

from _common import ROOT, build_workflow, load_config, load_tasks, write_jsonl

from src.fault_injection import FaultInjector, load_fault_specs
from src.recovery import STRATEGIES


def propagation_metrics(res) -> dict:
    """Which artifacts/actions are downstream of the injected fault."""
    tr = res.tracer
    faulty = [a.artifact_id for a in tr.artifacts.values() if a.faulty]
    affected = set()
    for fid in faulty:
        affected |= tr.downstream_of(fid)
    affected_actions = tr.actions_depending_on(set(faulty) | affected)
    return {
        "faulty_artifacts": faulty,
        "affected_artifacts": sorted(affected),
        "affected_stages": sorted({tr.artifacts[a].producer for a in affected}),
        "affected_actions": len(affected_actions),
        "unauthorized_actions": sum(not a.authorized for a in tr.actions),
        "irreversible_affected_actions": sum(not a.reversible for a in affected_actions),
        "tokens": sum(c.prompt_tokens + c.completion_tokens for c in tr.llm_calls),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tasks = load_tasks(cfg)
    specs = load_fault_specs(cfg["faults"])
    traces_dir = ROOT / cfg["output"]["traces_dir"]
    rows = []
    for rep in range(cfg["repeats"]):
        for task in tasks:
            for spec in specs:
                wf = build_workflow(cfg)
                inj = FaultInjector(spec)
                run_id = f"fault-{spec.id}-{task['task_id']}-r{rep}"
                res = wf.run(task, injector=inj, run_id=run_id)
                res.tracer.save(traces_dir)
                row = {
                    "run_id": run_id, "task_id": task["task_id"], "repeat": rep,
                    "fault_id": spec.id, "kind": spec.kind, "category": spec.category,
                    "stage": spec.stage, "fired": inj.fired, "success": res.success,
                    "latency_s": res.latency_s, "error": res.error,
                    **propagation_metrics(res),
                }
                # Recovery baselines (oracle fault identity)
                faulty_ids = row["faulty_artifacts"]
                for m in cfg["recovery"]["methods"]:
                    if not faulty_ids:
                        continue
                    rep_ = STRATEGIES[m]().recover(wf, task, res, faulty_ids[0])
                    row[f"{m}.success"] = rep_.recovered_success
                    row[f"{m}.recomputed"] = rep_.recomputed_stages
                    row[f"{m}.latency_s"] = rep_.recovery_latency_s
                    row[f"{m}.extra_tokens"] = rep_.extra_llm_tokens
                    row[f"{m}.irreversible"] = len(rep_.irreversible_actions)
                rows.append(row)
    out = ROOT / cfg["output"]["results_dir"] / "faults.jsonl"
    write_jsonl(rows, out)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
