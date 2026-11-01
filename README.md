# Failure Propagation and Recovery in Multi-Agent Systems

CSE291 project (Option 5). We study how faults in one agent's output propagate
through a researcher → planner → executor workflow, and whether
dependency tracking enables recovery that is cheaper than a full restart
while keeping execution correct and safe.

## Layout
```
configs/experiment.yaml   experiment settings: LLM backend, stages, fault schedules, recovery methods
src/
  tracing.py              Artifact / Message / ToolAction / LLMCall records + provenance graph
  environment.py          simulated shared state + tools (reversible / irreversible), snapshot/restore
  agents.py               pluggable LLM backends + Researcher / Planner / Executor / Verifier
  workflow.py             stage orchestration, fault hooks, task checkers
  fault_injection.py      FaultSpec + FaultInjector (incorrect / malicious / timeout / stale)
  recovery.py             FullRestart, VerifyAll (Phase 1), DependencyAwareInvalidation (Phase 2)
experiments/
  run_baseline.py         clean runs → results/baseline.jsonl
  run_faults.py           fault runs + recovery baselines → results/faults.jsonl
  analyze.py              aggregate tables/figures → reports/phase1/
data/tasks.json           tasks with programmatic success checks
results/                  raw outputs and traces (git-ignored)
reports/phase1/           Phase 1 analysis
```

## Setup
```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run (Phase 1)
```bash
python experiments/run_baseline.py
python experiments/run_faults.py
python experiments/analyze.py
```

## Status
Scaffold only. TODO: choose task domain and LLM backend, then implement
environment tools, agents, fault corruptions, checkers, and recovery baselines.
