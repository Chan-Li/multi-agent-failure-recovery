"""Brute-force schedule solver over a researcher-style fact sheet.

Used for (1) ground truth when generating tasks, (2) the mock planner, and
(3) the dependence check: re-solve on corrupted facts and see whether the
answer changes. Works on the *facts* format (see agents.py), so it can be run
on true facts (from the environment) or on faulty facts (from an artifact).
"""
from __future__ import annotations

from itertools import combinations, product
from typing import Any, Dict, List, Optional, Tuple


def to_min(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def _conflict(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if not set(a["days"]) & set(b["days"]):
        return False
    return to_min(a["start"]) < to_min(b["end"]) and to_min(b["start"]) < to_min(a["end"])


def section_index(facts: Dict[str, Any]) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    """sid -> (course code, section dict)."""
    out = {}
    for code, c in facts.get("courses", {}).items():
        for s in c.get("sections", []):
            out[s["section"]] = (code, s)
    return out


def schedule_problems(facts: Dict[str, Any], sections: List[str],
                      new: Optional[List[str]] = None) -> List[str]:
    """Rule violations for holding `sections` (seat check only for `new`)."""
    idx = section_index(facts)
    st = facts["student"]
    completed = set(st["completed"])
    problems, units, seen = [], 0, set()
    for sid in sections:
        if sid not in idx:
            problems.append(f"{sid}: unknown section")
            continue
        code, sec = idx[sid]
        c = facts["courses"][code]
        if code in seen:
            problems.append(f"{code}: two sections")
        seen.add(code)
        if code in completed:
            problems.append(f"{code}: already completed")
        missing = [p for p in c.get("prereqs", []) if p not in completed]
        if missing:
            problems.append(f"{code}: missing prereqs {missing}")
        if new and sid in new and sec.get("seats_open", 1) <= 0:
            problems.append(f"{sid}: full")
        units += c.get("units", 0)
    known = [s for s in sections if s in idx]
    for a, b in combinations(known, 2):
        if _conflict(idx[a][1], idx[b][1]):
            problems.append(f"time conflict {a} / {b}")
    if units > st["max_units"]:
        problems.append(f"units {units} > max {st['max_units']}")
    return problems


def solve(facts: Dict[str, Any], requirements: List[str],
          droppable: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Find the best plan covering every requirement.

    Objective (lexicographic): fewest drops, then alphabetical section ids.
    A requirement already covered by a kept enrolled course needs no action.
    Only courses in `droppable` may be dropped. Returns None if infeasible.
    """
    idx = section_index(facts)
    droppable = set(droppable or [])
    current = [s for s in facts["student"]["enrolled_sections"] if s in idx]
    can_drop = [s for s in current if idx[s][0] in droppable]
    best = None
    for k in range(len(can_drop) + 1):
        for drops in combinations(can_drop, k):
            keep = [s for s in current if s not in drops]
            covered = {r for s in keep for r in facts["courses"][idx[s][0]].get("satisfies", [])}
            todo = [r for r in requirements if r not in covered]
            options = []
            for r in todo:
                opts = sorted(sid for sid, (code, _) in idx.items()
                              if r in facts["courses"][code].get("satisfies", [])
                              and sid not in current)
                options.append(opts)
            for picks in product(*options):
                if len({idx[p][0] for p in picks}) < len(picks):
                    continue
                if schedule_problems(facts, keep + list(picks), new=list(picks)):
                    continue
                key = (len(drops), sorted(picks), sorted(drops))
                if best is None or key < best[0]:
                    best = (key, dict(zip(todo, picks)), list(drops), keep)
        if best is not None:
            break  # fewest drops found
    if best is None:
        return None
    _, assign, drops, keep = best
    return {
        "assign": assign,                     # requirement -> new section
        "drops": sorted(drops),
        "final_sections": sorted(keep + list(assign.values())),
    }


def plan_from_solution(facts: Dict[str, Any], requirements: List[str],
                       sol: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a solver result into the planner's JSON contract. Drops are
    attached to the subplan whose new section needs them (time conflict /
    unit cap), else to the first subplan."""
    idx = section_index(facts)
    subplans = []
    for r in requirements:
        sid = sol["assign"].get(r)
        subplans.append({"requirement": r, "course": idx[sid][0] if sid else None,
                         "enroll": sid, "drop": []})
    for d in sol["drops"]:
        target = next((sp for sp in subplans if sp["enroll"]
                       and _conflict(idx[sp["enroll"]][1], idx[d][1])), None)
        target = target or next((sp for sp in subplans if sp["enroll"]), subplans[0])
        target["drop"].append(d)
    return {"subplans": subplans, "other_actions": [], "rationale": "solver"}
