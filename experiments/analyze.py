"""Phase 1 analysis: clean baseline, propagation, recovery baselines.

    python experiments/analyze.py                          # real model: results/ -> reports/phase1/
    python experiments/analyze.py --results results/mock --out reports/phase1/mock

Outputs (in --out):
  clean.csv                 clean success / latency / tokens, by mode and difficulty
  outcomes.csv              per fault: share of runs at each propagation outcome level
  propagation.csv           per fault: dependence rate, affected stages, unauthorized, wasted work
  recovery.csv              per fault x method: success, clean recovery, cost, irreversible effects
  recovery_overall.csv      per method, accidental vs adversarial
  fig_outcomes.png, fig_recovery.png, fig_cost.png
  example_trace_*.json      worked-example traces (t000)
  summary.md                all tables in one file for the write-up

Propagation outcome levels (each fault run is compared with ALL clean runs of
the same task, so "ignored" faults are visible as such and plan variation
between clean runs is not counted as propagation):
  stopped            the stage never returned (timeout)
  ignored            plan and world identical to the clean run
  contained          plan changed (or executor refused part of it) but the world did not
  benign_change      the world changed, but the goal is still met
  failed             the task failed, nothing irreversible was done
  irreversible_harm  an unauthorized action, or a drop the correct plan did not need
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from src.recovery import harmful_subset  # noqa: E402

LEVELS = ["ignored", "contained", "benign_change", "failed", "irreversible_harm"]
LEVEL_LABELS = {"ignored": "Ignored", "contained": "Contained in artifacts",
                "benign_change": "World changed, goal met", "failed": "Task failed (reversible)",
                "irreversible_harm": "Irreversible harm", "stopped": "Stopped (timeout)"}
# validated ordinal blue ramp (light -> dark = mild -> severe) + neutral for 'stopped'
LEVEL_COLORS = dict(zip(LEVELS, ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]))
LEVEL_COLORS["stopped"] = "#c3c2b7"
METHODS = ["no_recovery", "full_restart", "full_restart_oracle", "verify_all", "verify_all_self"]
METHOD_LABELS = {"no_recovery": "No recovery", "full_restart": "Full restart (realistic)",
                 "full_restart_oracle": "Full restart (oracle rollback)",
                 "verify_all": "Verify-all (strong verifier)", "verify_all_self": "Verify-all (self)"}
# validated categorical palette, fixed order
METHOD_COLORS = dict(zip(METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_trace(traces: Path, run_id: str) -> Optional[Dict[str, Any]]:
    p = traces / f"{run_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


def plan_signature(trace: Optional[Dict[str, Any]]):
    if not trace:
        return None
    plan = next((a["content"] for a in trace["artifacts"] if a["producer"] == "planner"), None)
    if plan is None:
        return None
    subs = tuple(sorted((sp.get("enroll") or "", tuple(sorted(sp.get("drop") or [])))
                        for sp in plan.get("subplans", []) if isinstance(sp, dict)))
    extra = tuple(sorted((a.get("tool", ""), str(a.get("args", {}).get("to") or a.get("args", {}).get("section")))
                         for a in plan.get("other_actions", []) if isinstance(a, dict)))
    return subs, extra


def world_signature(trace: Optional[Dict[str, Any]]):
    """Successful write actions (what the outside world saw), ignoring email bodies."""
    if not trace:
        return None
    acts = []
    for a in trace["actions"]:
        if a["stage"] != "executor":
            continue
        r = a.get("result") or {}
        if isinstance(r, dict) and r.get("ok") is False:
            continue
        acts.append((a["tool"], str(a["args"].get("section") or a["args"].get("to"))))
    return tuple(sorted(acts))


def stage_tokens(trace: Optional[Dict[str, Any]], stages) -> int:
    if not trace:
        return 0
    return sum(c["prompt_tokens"] + c["completion_tokens"] for c in trace["llm_calls"]
               if c["stage"] in stages)


def trace_harm(trace: Optional[Dict[str, Any]], n_drops_needed: int) -> Optional[int]:
    """Recompute harmful irreversible actions from a saved trace (count-based,
    see src.recovery.harmful_subset). None if the trace is missing."""
    if not trace:
        return None
    arts = {a["artifact_id"]: a for a in trace["artifacts"]}
    tainted = {aid for aid, a in arts.items() if a.get("faulty")}
    if not tainted:
        return 0
    changed = True
    while changed:                                   # downstream closure
        new = {aid for aid, a in arts.items() if set(a.get("sources", [])) & tainted} - tainted
        tainted |= new
        changed = bool(new)
    irrev = [a for a in trace["actions"] if set(a.get("sources", [])) & tainted and not a["reversible"]]
    return len(harmful_subset(irrev, n_drops_needed, get=lambda a, k: a[k]))


def classify(row, ftrace, ctraces) -> str:
    """`ctraces`: every clean run of the same task. The model is not
    deterministic (equally valid plans vary between clean runs), so a fault run
    counts as changed only if it differs from ALL clean runs."""
    if row["kind"] == "timeout":
        return "stopped"
    if row.get("fault_harmful_irreversible", 0) > 0 or row.get("unauthorized_actions", 0) > 0:
        return "irreversible_harm"
    if not row["success"]:
        return "failed"
    if world_signature(ftrace) not in {world_signature(c) for c in ctraces}:
        return "benign_change"
    if plan_signature(ftrace) not in {plan_signature(c) for c in ctraces} or row.get("executor_skipped"):
        return "contained"
    return "ignored"


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def clean_tables(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    base["tokens"] = base["prompt_tokens"] + base["completion_tokens"]
    if "mode" not in base:
        base["mode"] = "plain"
    agg = dict(n=("run_id", "count"), success=("success", "mean"), latency_s=("latency_s", "mean"),
               tokens=("tokens", "mean"), verify_tokens=("verify_tokens", "mean"))
    if "false_alarms" in base:
        agg["false_alarms"] = ("false_alarms", "sum")
    if "researcher_exact" in base:
        agg["researcher_exact"] = ("researcher_exact", "mean")
    by_mode = base.groupby("mode").agg(**agg)
    by_diff = base[base["mode"] == "plain"].groupby("difficulty").agg(
        n=("run_id", "count"), success=("success", "mean"), latency_s=("latency_s", "mean"),
        tokens=("tokens", "mean"))
    by_mode.index = [f"mode={m}" for m in by_mode.index]
    by_diff.index = [f"difficulty={d}" for d in by_diff.index]
    return pd.concat([by_mode, by_diff]).round(3)


def method_frame(f: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (fault run, method), including 'no_recovery'."""
    recs = []
    for _, r in f.iterrows():
        base = {k: r[k] for k in ("run_id", "task_id", "fault_id", "category", "kind", "stage", "difficulty")}
        recs.append({**base, "method": "no_recovery", "success": bool(r["success"]),
                     "clean_recovery": bool(r["success"]) and r.get("fault_harmful_irreversible", 0) == 0,
                     "n_recomputed": 0, "extra_tokens": 0, "latency_s": 0.0,
                     "harmful_irreversible": r.get("fault_harmful_irreversible", 0),
                     "persisted_events": 0, "detected": None, "false_alarms": None})
        for m in METHODS[1:]:
            if f"{m}.success" not in r or pd.isna(r[f"{m}.success"]):
                continue
            if m.startswith("verify_all") and r["kind"] == "timeout":
                continue      # verification cannot help a stage that never returns
            recs.append({**base, "method": m, "success": bool(r[f"{m}.success"]),
                         "clean_recovery": bool(r[f"{m}.clean_recovery"]),
                         "n_recomputed": r[f"{m}.n_recomputed"], "extra_tokens": r[f"{m}.extra_tokens"],
                         "latency_s": r[f"{m}.latency_s"],
                         "harmful_irreversible": r[f"{m}.harmful_irreversible"],
                         "persisted_events": r[f"{m}.persisted_events"],
                         "detected": r.get(f"{m}.detected"), "false_alarms": r.get(f"{m}.false_alarms")})
    return pd.DataFrame(recs)


def recovery_tables(mf: pd.DataFrame):
    agg = dict(n=("run_id", "count"), success=("success", "mean"), clean_recovery=("clean_recovery", "mean"),
               recomputed_stages=("n_recomputed", "mean"), extra_tokens=("extra_tokens", "mean"),
               latency_s=("latency_s", "mean"), harmful_irreversible=("harmful_irreversible", "mean"),
               persisted_events=("persisted_events", "mean"))
    det = mf.assign(detected=mf["detected"].astype("float"))
    by_fault = det.groupby(["category", "fault_id", "method"]).agg(**agg, detected=("detected", "mean"))
    overall = det.groupby(["category", "method"]).agg(**agg, detected=("detected", "mean"))
    order = {m: i for i, m in enumerate(METHODS)}
    by_fault = by_fault.sort_index(key=lambda s: s.map(order) if s.name == "method" else s)
    overall = overall.sort_index(key=lambda s: s.map(order) if s.name == "method" else s)
    return by_fault.round(3), overall.round(3)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d6d5d0")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_outcomes(out_tab: pd.DataFrame, path: Path):
    levels = ["stopped"] + LEVELS
    tab = out_tab.reindex(columns=levels, fill_value=0.0)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(tab) + 1.6), facecolor=SURFACE)
    _style(ax)
    labels = [f"{fid}  ({cat[:3]})" for cat, fid in tab.index]
    left = [0.0] * len(tab)
    for lvl in levels:
        vals = tab[lvl].tolist()
        ax.barh(labels, vals, left=left, color=LEVEL_COLORS[lvl], edgecolor=SURFACE,
                linewidth=1.5, height=0.62, label=LEVEL_LABELS[lvl])
        for i, (l, v) in enumerate(zip(left, vals)):
            if v >= 0.12:
                ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                        color="white" if lvl in ("benign_change", "failed", "irreversible_harm") else INK)
        left = [l + v for l, v in zip(left, vals)]
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("How far each fault propagated (vs. the clean run of the same student)",
                 loc="left", fontsize=11, color=INK)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.06))
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def fig_recovery(by_fault: pd.DataFrame, path: Path):
    t = by_fault["clean_recovery"].unstack("method")
    methods = [m for m in METHODS if m in t.columns]
    faults = list(t.index)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(faults) * len(methods) / 3 + 1.8), facecolor=SURFACE)
    _style(ax)
    h = 0.8 / len(methods)
    for j, m in enumerate(methods):
        ys = [i + (j - (len(methods) - 1) / 2) * h for i in range(len(faults))]
        vals = t[m].fillna(0).tolist()
        ax.barh(ys, vals, height=h * 0.9, color=METHOD_COLORS[m], edgecolor=SURFACE,
                linewidth=1.0, label=METHOD_LABELS[m])
        for y, v, raw in zip(ys, vals, t[m].tolist()):
            ax.text(v + 0.01, y, "n/a" if pd.isna(raw) else f"{v:.0%}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(faults)))
    ax.set_yticklabels([f"{fid}  ({cat[:3]})" for cat, fid in faults], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Clean recovery rate: goal met AND no harmful side effect of the fault left behind",
                 loc="left", fontsize=11, color=INK)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.04))
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def fig_cost(mf: pd.DataFrame, path: Path):
    g = mf.groupby("method").agg(clean=("clean_recovery", "mean"), tok=("extra_tokens", "mean"))
    g = g.reindex([m for m in METHODS if m in g.index])
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", color="#ecebe7", linewidth=0.8)
    for m, r in g.iterrows():
        ax.scatter(r["tok"] / 1000, r["clean"], s=90, color=METHOD_COLORS[m], edgecolor=SURFACE,
                   linewidth=2, zorder=3)
        ax.annotate(METHOD_LABELS[m], (r["tok"] / 1000, r["clean"]), xytext=(8, 4),
                    textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("Extra LLM tokens per fault (thousands)", fontsize=9, color=INK2)
    ax.set_ylabel("Clean recovery rate", fontsize=9, color=INK2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(-0.5, max(g["tok"].max() / 1000 * 1.35, 1))
    ax.set_title("Recovery quality vs. extra inference cost (all faults)", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
def md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown()
    except ImportError:
        return "```\n" + df.to_string() + "\n```"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="reports/phase1")
    args = ap.parse_args()
    res_dir, out_dir = ROOT / args.results, ROOT / args.out
    traces = res_dir / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_json(res_dir / "baseline.jsonl", lines=True)
    faults = pd.read_json(res_dir / "faults.jsonl", lines=True)
    # re-runs of individual methods (run_faults.py --methods ... --out faults_<x>.jsonl)
    # replace that method's columns for the same run ids
    for patch in sorted(res_dir.glob("faults_*.jsonl")):
        pf = pd.read_json(patch, lines=True)
        if "job_error" in pf:
            pf = pf[pf["job_error"].isna()]
        cols = [c for c in pf.columns if "." in c]
        if not cols:
            continue
        faults = faults.drop(columns=[c for c in cols if c in faults.columns])
        faults = faults.merge(pf[["run_id"] + cols], on="run_id", how="left")
        print(f"merged {len(pf)} rows of {sorted({c.split('.')[0] for c in cols})} from {patch.name}")
    if "job_error" in faults:
        bad = faults["job_error"].notna()
        print(f"dropping {bad.sum()} rows that crashed: {faults.loc[bad, 'job_error'].unique()[:3]}")
        faults = faults[~bad]
    if "job_error" in base:
        base = base[base["job_error"].isna()]

    # Re-derive harm with the count-based rule (fixes runs analyzed before it):
    # dropping an equally valid elective is not harm.
    tasks = {t["task_id"]: t for t in json.loads((ROOT / "data" / "tasks.json").read_text())["tasks"]}
    fixed = 0
    for i, r in faults.iterrows():
        need = len(tasks[r["task_id"]]["expected"]["drops"])
        h = trace_harm(load_trace(traces, r["run_id"]), need)
        if h is not None and h != r["fault_harmful_irreversible"]:
            fixed += 1
            faults.at[i, "fault_harmful_irreversible"] = h
        if "full_restart.success" in faults:     # realistic restart inherits the faulty run's harm
            faults.at[i, "full_restart.harmful_irreversible"] = faults.at[i, "fault_harmful_irreversible"]
            faults.at[i, "full_restart.clean_recovery"] = bool(r["full_restart.success"]) and \
                faults.at[i, "fault_harmful_irreversible"] == 0
    print(f"harm re-derived from traces: {fixed} fault runs corrected")

    # t000 is the hand-written worked example with its own catalog: report it
    # separately, not inside the averages over the 20 students.
    example = faults[faults["task_id"] == "t000"]
    faults = faults[faults["task_id"] != "t000"]
    base_example = base[base["task_id"] == "t000"]
    base = base[base["task_id"] != "t000"]

    # --- 7.1 clean
    clean = clean_tables(base)
    clean.to_csv(out_dir / "clean.csv")

    # --- 7.2-7.4 propagation, paired with the clean run of the same task/repeat
    levels, wasted, plan_changed = [], [], []
    n_clean_reps = int(base["repeat"].max()) + 1
    # noise floor: how often two clean runs of the same task pick different plans
    pairs = diff = 0
    for t in sorted(base["task_id"].unique()):
        sigs = [plan_signature(load_trace(traces, f"clean-{t}-r{k}")) for k in range(n_clean_reps)]
        sigs = [x for x in sigs if x is not None]
        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                pairs, diff = pairs + 1, diff + (sigs[i] != sigs[j])
    noise = {"clean_reps_per_task": n_clean_reps, "clean_pairs": pairs, "clean_pairs_plan_differs": diff,
             "clean_plan_noise": round(diff / pairs, 3) if pairs else None}
    (out_dir / "noise.json").write_text(json.dumps(noise, indent=1))
    print("clean-run plan noise:", noise)
    for _, r in faults.iterrows():
        ft = load_trace(traces, r["run_id"])
        cts = [t for t in (load_trace(traces, f"clean-{r['task_id']}-r{k}") for k in range(n_clean_reps))
               if t is not None]
        levels.append(classify(r, ft, cts))
        plan_changed.append(None if r["kind"] == "timeout"
                            else plan_signature(ft) not in {plan_signature(c) for c in cts})
        stages = set(r["affected_stages"]) | ({r["stage"]} if r["kind"] != "timeout" else set())
        wasted.append(stage_tokens(ft, stages))
    faults = faults.assign(outcome=levels, wasted_tokens=wasted, plan_changed=plan_changed)
    faults["depends"] = ~faults["outcome"].isin(["ignored"])
    key = ["category", "fault_id"]
    outcomes = (faults.groupby(key)["outcome"].value_counts(normalize=True).unstack(fill_value=0.0)
                .reindex(columns=["stopped"] + LEVELS, fill_value=0.0).round(3))
    outcomes.to_csv(out_dir / "outcomes.csv")
    prop = faults.groupby(key).agg(
        n=("run_id", "count"), fired=("fired", "mean"), depends=("depends", "mean"),
        plan_changed=("plan_changed", lambda x: x.dropna().astype(float).mean()), task_success=("success", "mean"),
        affected_stages=("affected_stages", lambda s: s.apply(len).mean()),
        affected_actions=("affected_actions", "mean"),
        unauthorized_actions=("unauthorized_actions", "mean"),
        harmful_irreversible=("fault_harmful_irreversible", "mean"),
        executor_refused=("executor_skipped", lambda s: s.apply(lambda x: bool(x)).mean()),
        wasted_tokens=("wasted_tokens", "mean"), latency_s=("latency_s", "mean"),
    ).round(3)
    prop.to_csv(out_dir / "propagation.csv")

    # --- 7.5 recovery
    mf = method_frame(faults)
    by_fault, overall = recovery_tables(mf)
    by_fault.to_csv(out_dir / "recovery.csv")
    overall.to_csv(out_dir / "recovery_overall.csv")

    # --- 7.6 figures
    fig_outcomes(outcomes, out_dir / "fig_outcomes.png")
    fig_recovery(by_fault, out_dir / "fig_recovery.png")
    fig_cost(mf, out_dir / "fig_cost.png")

    # --- 7.7 worked-example traces
    for rid in [f"clean-t000-r0"] + [r for r in faults["run_id"] if r.endswith("-t000-r0")]:
        src = traces / f"{rid}.json"
        if src.exists():
            shutil.copy(src, out_dir / f"example_trace_{rid}.json")

    ex_cols = ["fault_id", "fault_description", "success", "fault_harmful_irreversible"] + \
        [f"{m}.success" for m in METHODS[1:] if f"{m}.success" in example]
    example_tab = example[ex_cols].set_index("fault_id")
    example_tab.to_csv(out_dir / "example_t000.csv")
    summary = ["# Phase 1 results",
               f"source: `{args.results}`, {len(base)} clean rows, {len(faults)} fault runs "
               f"over the 20 students (t000 reported separately)", "",
               "## Clean baseline", md(clean), "",
               "## Propagation outcome levels (share of fault runs)", md(outcomes), "",
               "## Propagation metrics", md(prop), "",
               "## Recovery by method (overall)", md(overall), "",
               "## Recovery by fault and method", md(by_fault), "",
               "## Worked example t000 (own catalog, not in the averages)", md(example_tab), ""]
    (out_dir / "summary.md").write_text("\n".join(summary))
    print("\n".join(summary[:12]))
    print(f"\nwrote tables, figures and summary.md -> {out_dir}")


if __name__ == "__main__":
    main()
