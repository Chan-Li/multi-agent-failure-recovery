"""Aggregate baseline/fault results into tables and figures for reports/phase1."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="reports/phase1")
    args = ap.parse_args()
    res_dir, out_dir = ROOT / args.results, ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_json(res_dir / "baseline.jsonl", lines=True)
    faults = pd.read_json(res_dir / "faults.jsonl", lines=True)

    summary = {"clean_success_rate": base["success"].mean(),
               "clean_latency_s": base["latency_s"].mean()}
    print(summary)

    # Only count runs where the fault actually fired; separate accidental vs adversarial.
    fired = faults[faults["fired"]]
    by = fired.groupby(["category", "kind", "stage"]).agg(
        n=("run_id", "count"),
        success=("success", "mean"),
        affected_actions=("affected_actions", "mean"),
        unauthorized=("unauthorized_actions", "mean"),
        latency_s=("latency_s", "mean"),
    )
    print(by)
    by.to_csv(out_dir / "propagation_by_fault.csv")
    # TODO: dependence check (did the output actually change vs clean run?),
    #       wasted work, recovery baseline comparison, figures.


if __name__ == "__main__":
    main()
