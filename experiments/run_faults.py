"""Phase 1: run each fault schedule, measure propagation, then apply the
baseline recovery methods (full restart, verify-all) under the same schedule."""
from __future__ import annotations

import argparse
import copy

from _common import ROOT, build_strategy, build_workflow, load_config, load_tasks, write_jsonl

from src.fault_injection import FaultInjector, load_fault_specs
from src.recovery import fault_side_effects
from src.workflow import Workflow


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
    ap.add_argument("--no-recovery", action="store_true", help="propagation only (skip recovery baselines)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tasks = load_tasks(cfg)[:args.limit]
    specs = load_fault_specs(cfg["faults"])
    traces_dir = ROOT / cfg["output"]["traces_dir"]
    strategies = {m: build_strategy(cfg, m) for m in cfg["recovery"]["methods"]}
    rows, skipped = [], 0
    for rep in range(cfg["repeats"]):
        for task in tasks:
            for spec in specs:
                if spec.kind == "none" or spec.id not in task.get("valid_faults", []):
                    skipped += spec.kind != "none"
                    continue          # no-fault runs live in baseline.jsonl
                wf = build_workflow(cfg)
                inj = FaultInjector(spec, task)
                run_id = f"fault-{spec.id}-{task['task_id']}-r{rep}"
                res = wf.run(task, injector=inj, run_id=run_id)
                res.tracer.save(traces_dir)
                row = {
                    "run_id": run_id, "task_id": task["task_id"], "repeat": rep,
                    "fault_id": spec.id, "kind": spec.kind, "category": spec.category,
                    "stage": spec.stage, "fault_type": spec.type, "difficulty": task.get("difficulty"),
                    "fault_description": inj.description,
                    "mock_harmful": task["faults"][spec.id]["harmful"],
                    "fired": inj.fired, "success": res.success,
                    "check_failures": res.metrics["check_failures"],
                    "final_sections": res.metrics["final_state"]["sections"],
                    "latency_s": res.latency_s, "error": res.error,
                    **propagation_metrics(res),
                }
                irrev, harmful = fault_side_effects(res, task, row["faulty_artifacts"][0]
                                                    if row["faulty_artifacts"] else None)
                row["fault_irreversible"] = len(irrev)
                row["fault_harmful_irreversible"] = len(harmful)
                # Recovery baselines (oracle fault identity). Each method starts
                # from its own copy of the world exactly as the faulty run left it.
                faulty_id = row["faulty_artifacts"][0] if row["faulty_artifacts"] else None
                for m in ([] if args.no_recovery else cfg["recovery"]["methods"]):
                    if m == "verify_all" and spec.kind == "timeout":
                        continue      # verification cannot help a stage that never returns
                    wf_m = Workflow(wf.agents, copy.deepcopy(wf.env))
                    rep_ = strategies[m].recover(wf_m, task, res, faulty_id, spec)
                    rep_.final.tracer.save(traces_dir)
                    row.update(rep_.to_row(m))
                rows.append(row)
    out = ROOT / cfg["output"]["results_dir"] / "faults.jsonl"
    write_jsonl(rows, out)
    print(f"wrote {len(rows)} rows -> {out} (skipped {skipped} task x fault pairs with no dependence)")


if __name__ == "__main__":
    main()
