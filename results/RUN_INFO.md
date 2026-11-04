# Phase 1 raw results: what is here and how it was produced

All real-model data behind `reports/phase1/` and the Phase 1 report.
Model API: UCSD OpenAI-compatible endpoint (TritonAI). Agents (researcher,
planner, executor) and the self verifier: `api-glm-5.3-flash`; strong
verifier: `api-glm-5.3`; temperature 0 everywhere. Total API cost about $9.3.

## Files

| File | Rows | What it is |
|---|---|---|
| `baseline.jsonl` | 105 | Clean runs (no fault): 21 tasks x 3 plain repeats, plus 1 run per task with each verifier (strong, self) |
| `faults.jsonl` | 164 | One row per (task, fault) pair: the faulty run without recovery, plus the full restart (realistic, oracle) columns. Its `verify_all*` columns are **superseded** (see below) |
| `faults_verify.jsonl` | 143 | Always-verify (strong and self) for every non-timeout pair, rerun with a 16k verifier budget. `analyze.py` uses these columns in place of the ones in `faults.jsonl` |
| `archive.tar.gz` | | All traces plus two side experiments (below). Unpack with `tar -xzf results/archive.tar.gz -C results` |

Inside `archive.tar.gz`:

| Path | What it is |
|---|---|
| `traces/` | 883 JSON traces, one per run: every artifact, message, tool action and LLM call (token counts). Needed by `analyze.py` |
| `excluded_check/` | `faults.jsonl` + 30 traces: the 4 (task, fault) pairs the mock filter dropped, run on the real model as a check (not in the main tables) |
| `obsolete/` | The first strong-verifier clean runs at a 4096-token budget (21 rows, 42 traces). 6 of 42 verifier calls ran out of budget and returned empty replies. **Not used** in any result; kept for transparency |

Mock-backend results (`results/mock/`, about 135 MB) are not committed: they
are deterministic and free to regenerate (see the README).

## Run log

| When (local time) | What | Command | Budgets (output tokens) |
|---|---|---|---|
| 11-02 10:29 to 10:40 | Clean plain runs, repeat r0 | `run_baseline.py --verify` | researcher 12000, others 4096 |
| 11-02 10:43 to 10:50 | Clean runs with self verifier | (same command) | self verifier 4096 (largest output 809, never near the cap) |
| 11-02 10:43 to 15:04 | Fault runs + full restarts (+ first always-verify, superseded) | `run_faults.py --workers 8` | as above; verifier 4096 |
| 11-02 20:28 to 21:18 | Always-verify rerun, strong and self | `run_faults.py --methods verify_all,verify_all_self --out faults_verify.jsonl --workers 8` | verifiers 16000 |
| 11-03 10:08 to 10:12 | Clean plain runs, repeats r1 and r2 (noise floor) | `run_baseline.py --repeats 3` | as r0 |
| 11-03 11:15 to 11:30 | Clean runs with strong verifier, rerun | `run_baseline.py --verify` | strong verifier 16000 |
| 11-03 14:40 to 14:44 | Excluded-pair check | `run_faults.py --config configs/excluded_check.yaml --repeats 1 --workers 6` | as the main runs, verifiers 16000 |

The scripts are resumable: rerunning a command skips run ids already in the
output file, which is how batches were added.

## Why the always-verify columns were rerun

In the first fault run the strong verifier (a reasoning model) sometimes used
its whole 4096-token budget on hidden reasoning and returned an empty reply,
which was then scored as "no problem found". Trace files of the strong and
self variants also shared a name. Both were fixed (commit `4bd93bf`) and the
always-verify methods were rerun; full restart and the faulty runs were not
affected and were not rerun.

## Known caveats

- **Latency.** The shared API stalled for long periods during the fault runs,
  so `latency_s` includes queueing. Compare methods by tokens.
- **Single repeat.** Each fault run is one sample per pair; the model is not
  deterministic at temperature 0 (33% of clean-run pairs choose different
  valid plans), so fault runs are compared with all 3 clean runs of a task.
- **Two planner budget hits.** In `fault-researcher-wrong-prereq-t001-r0` and
  `fault-researcher-wrong-units-t018-r0` the planner used its whole 4096
  tokens and returned no parseable plan (row field `parse_failures`).
- **What traces do not contain.** Phase 1 traces record token counts but not
  the raw reply text or the API stop reason, and a verifier-rejected artifact
  is replaced by its regeneration. Later code records all three.
- **Harm column.** `fault_harmful_irreversible` in `faults.jsonl` was computed
  by matching section ids; `analyze.py` recomputes it from the traces by
  counting drops (8 runs differ). Use the analyzed tables, not the raw column.

## Key row fields

`faults.jsonl` / `faults_verify.jsonl` (prefix `<method>.` for recovery columns,
methods `full_restart`, `full_restart_oracle`, `verify_all`, `verify_all_self`):

| Field | Meaning |
|---|---|
| `run_id`, `task_id`, `fault_id`, `category`, `stage` | which pair; `category` is accidental or adversarial |
| `fault_description` | the concrete corruption for this task |
| `success`, `check_failures` | goal met by the final world state, and why not |
| `affected_actions`, `unauthorized_actions`, `parse_failures` | propagation counts; agents whose reply did not parse |
| `<m>.success`, `<m>.clean_recovery` | recovered: goal met (and no harmful fault effect left) |
| `<m>.extra_tokens`, `<m>.n_recomputed`, `<m>.persisted_events` | recovery cost; world events a restore could not undo |
| `<m>.detected`, `<m>.verdicts` | always-verify: was the faulty artifact rejected; every verdict with its issues |

`baseline.jsonl`: `mode` (plain, verify_all, verify_all_self), `repeat`,
`success`, token counts, `researcher_exact` (fact sheet equals the registrar),
`false_alarms` (number of verifier rejections; see the report on why these
were real planner errors on fault runs).
