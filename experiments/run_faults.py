"""Phase 1: run each fault schedule, measure propagation, then apply the
baseline recovery methods (full restart, verify-all) under the same schedule."""
from __future__ import annotations

import argparse
import copy

from _common import ROOT, build_strategy, build_workflow, load_config, load_tasks, run_jobs

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


def run_one(job, cfg, strategies, traces_dir, no_recovery) -> dict:
    task, spec, rep, run_id = job["task"], job["spec"], job["repeat"], job["run_id"]
    wf = build_workflow(cfg)
    inj = FaultInjector(spec, task)
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
        "parse_failures": [a.producer for a in res.outputs.values() if a.meta.get("parse_failed")],
        "executor_skipped": (res.outputs["executor"].content.get("skipped")
                             if "executor" in res.outputs else None),
        **propagation_metrics(res),
    }
    faulty_id = row["faulty_artifacts"][0] if row["faulty_artifacts"] else None
    irrev, harmful = fault_side_effects(res, task, faulty_id)
    row["fault_irreversible"] = len(irrev)
    row["fault_harmful_irreversible"] = len(harmful)
    # Recovery baselines (oracle fault identity). Each method starts from its
    # own copy of the world exactly as the faulty run left it.
    for m, strategy in ({} if no_recovery else strategies).items():
        if m == "verify_all" and spec.kind == "timeout":
            continue          # verification cannot help a stage that never returns
        wf_m = Workflow(wf.agents, copy.deepcopy(wf.env))
        rep_ = strategy.recover(wf_m, task, res, faulty_id, spec)
        rep_.final.tracer.save(traces_dir)
        row.update(rep_.to_row(m))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--no-recovery", action="store_true", help="propagation only (skip recovery baselines)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    ap.add_argument("--repeats", type=int, default=None, help="override cfg repeats")
    ap.add_argument("--faults", default=None, help="comma-separated fault ids to run (default all)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--fresh", action="store_true", help="discard existing results instead of resuming")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tasks = load_tasks(cfg)[:args.limit]
    specs = [s for s in load_fault_specs(cfg["faults"]) if s.kind != "none"
             and (args.faults is None or s.id in args.faults.split(","))]
    traces_dir = ROOT / cfg["output"]["traces_dir"]
    strategies = {m: build_strategy(cfg, m) for m in cfg["recovery"]["methods"]}
    jobs, skipped = [], 0
    for rep in range(args.repeats or cfg["repeats"]):
        for task in tasks:
            for spec in specs:
                if spec.id not in task.get("valid_faults", []):
                    skipped += 1
                    continue
                jobs.append({"run_id": f"fault-{spec.id}-{task['task_id']}-r{rep}",
                             "task": task, "spec": spec, "repeat": rep})
    print(f"skipping {skipped} task x fault pairs with no dependence")
    out = ROOT / cfg["output"]["results_dir"] / "faults.jsonl"
    n = run_jobs(jobs, lambda j: run_one(j, cfg, strategies, traces_dir, args.no_recovery),
                 out, workers=args.workers, fresh=args.fresh)
    print(f"wrote {n} new rows -> {out}")


if __name__ == "__main__":
    main()
