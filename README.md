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
uv pip install -r requirements.lock      # exact versions used for Phase 1
# (requirements.txt has the loose ranges)
```
The real-model runs need `OPENAI_API_KEY` and `OPENAI_BASE_URL` in a local
`.env` (git-ignored). `configs/mock.yaml` runs everything with the
deterministic mock backend at no cost.

## Run (Phase 1)
```bash
python experiments/gen_tasks.py --seed 0                      # tasks + dependent fault patches
python experiments/run_baseline.py --verify --workers 6       # clean runs, plain + both verifiers
python experiments/run_baseline.py --repeats 3 --workers 6    # 2 more clean runs per task (noise floor)
python experiments/run_faults.py --workers 8                  # fault runs + recovery baselines (resumable)
python experiments/run_faults.py --methods verify_all,verify_all_self \
    --out faults_verify.jsonl --workers 8                     # verify-only rerun, merged by analyze.py
python experiments/analyze.py                                 # tables, figures, summary -> reports/phase1/
pytest -q
```
Mock version: add `--config configs/mock.yaml` to the run scripts and
`--results results/mock --out reports/phase1/mock` to `analyze.py`.
Raw results live in `results/` (git-ignored, archived separately); see
`results/RUN_INFO.md` for how and when they were produced.

## Status
Phase 1 complete: workflow, fault injection, tracing, full-restart and
always-verify baselines, real-model results and report
(`reports/phase1/report/main.pdf`). Phase 2 (dependency-aware invalidation,
`src/recovery.py: DependencyAwareInvalidation`) is not implemented yet.
