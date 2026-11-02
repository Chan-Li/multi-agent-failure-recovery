"""Phase 1: run the clean (no-fault) workflow and record clean task success."""
from __future__ import annotations

import argparse

from _common import ROOT, build_agent, build_workflow, load_config, load_tasks, run_jobs

from src.agents import Verifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--verify", action="store_true",
                    help="also run each task with always-verify (false alarms + overhead on clean input)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N tasks")
    ap.add_argument("--repeats", type=int, default=None, help="override cfg repeats")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--fresh", action="store_true", help="discard existing results instead of resuming")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tasks = load_tasks(cfg)[:args.limit]
    verifier = None
    if args.verify:
        v = build_agent(cfg, "verifier")
        verifier = Verifier(v.llm, tool_access=cfg["recovery"].get("verifier_tool_access", True))
    traces_dir = ROOT / cfg["output"]["traces_dir"]

    def run_one(job):
        task, rep, mode = job["task"], job["repeat"], job["mode"]
        wf = build_workflow(cfg)
        res = wf.run(task, injector=None, run_id=job["run_id"],
                     verifier=verifier if mode == "verify_all" else None)
        res.tracer.save(traces_dir)
        calls = res.tracer.llm_calls
        row = {
            "run_id": res.run_id, "task_id": res.task_id, "repeat": rep, "mode": mode,
            "difficulty": task.get("difficulty"), "fault_id": "none",
            "success": res.success, "check_failures": res.metrics["check_failures"],
            "final_sections": res.metrics["final_state"]["sections"],
            "latency_s": res.latency_s, "error": res.error,
            "prompt_tokens": sum(c.prompt_tokens for c in calls),
            "completion_tokens": sum(c.completion_tokens for c in calls),
            "verify_tokens": sum(c.prompt_tokens + c.completion_tokens for c in calls
                                 if c.purpose == "verify"),
            "parse_failures": [a.producer for a in res.outputs.values() if a.meta.get("parse_failed")],
            "researcher_exact": (res.outputs["researcher"].content == wf.env.__class__(
                task["inputs"]["scenario"]).true_facts()) if "researcher" in res.outputs else None,
        }
        if mode == "verify_all":
            row["false_alarms"] = sum(not v["ok"] for v in res.metrics["verdicts"])
            row["regenerated"] = res.metrics["regenerated"]
        return row

    jobs = []
    for rep in range(args.repeats or cfg["repeats"]):
        for task in tasks:
            for mode in (["plain", "verify_all"] if verifier else ["plain"]):
                rid = f"clean-{task['task_id']}-r{rep}" + ("" if mode == "plain" else "-verify")
                jobs.append({"run_id": rid, "task": task, "repeat": rep, "mode": mode})
    out = ROOT / cfg["output"]["results_dir"] / "baseline.jsonl"
    n = run_jobs(jobs, run_one, out, workers=args.workers, fresh=args.fresh)
    print(f"wrote {n} new rows -> {out}")


if __name__ == "__main__":
    main()
