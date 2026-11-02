"""Phase 1: run the clean (no-fault) workflow and record clean task success."""
from __future__ import annotations

import argparse

from _common import ROOT, build_workflow, load_config, load_tasks, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tasks = load_tasks(cfg)
    traces_dir = ROOT / cfg["output"]["traces_dir"]
    rows = []
    for rep in range(cfg["repeats"]):
        for task in tasks:
            wf = build_workflow(cfg)
            res = wf.run(task, injector=None, run_id=f"clean-{task['task_id']}-r{rep}")
            res.tracer.save(traces_dir)
            calls = res.tracer.llm_calls
            rows.append({
                "run_id": res.run_id, "task_id": res.task_id, "repeat": rep,
                "difficulty": task.get("difficulty"), "fault_id": "none",
                "success": res.success, "check_failures": res.metrics["check_failures"],
                "final_sections": res.metrics["final_state"]["sections"],
                "latency_s": res.latency_s, "error": res.error,
                "prompt_tokens": sum(c.prompt_tokens for c in calls),
                "completion_tokens": sum(c.completion_tokens for c in calls),
                "parse_failures": [a.producer for a in res.outputs.values() if a.meta.get("parse_failed")],
            })
    out = ROOT / cfg["output"]["results_dir"] / "baseline.jsonl"
    write_jsonl(rows, out)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
