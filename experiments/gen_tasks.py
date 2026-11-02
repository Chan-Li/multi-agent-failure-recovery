"""Generate course-enrollment tasks with a seeded RNG -> data/tasks.json.

Task format (consumed by Workflow + the `enroll_goal` checker)::

    {"task_id", "difficulty": easy|medium|hard, "check": "enroll_goal",
     "goal": natural-language request (what the agents read),
     "requirements": [req, ...],
     "constraints": {"droppable": [course codes], "email_to": address},
     "inputs": {"scenario": <CourseEnrollmentEnv scenario>},
     "expected": {"assign": {req: section}, "drops": [section], "final_sections": [...]}}

Catalogs are fictional (no model prior knowledge). Ground truth comes from the
brute-force solver on the TRUE facts. Difficulty:
  easy    1 requirement, nothing needs dropping
  medium  2 requirements, prereq chains of depth <= 2
  hard    3 requirements, unit cap or a time conflict forces exactly one drop
Competitors arrive at t >= 3 (after the clean executor, which runs at t = 2),
so they never break a clean run but do take seats while recovery is running.

    python experiments/gen_tasks.py [--seed 0] [--easy 6 --medium 8 --hard 6]
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.environment import DEFAULT_SCENARIO, CourseEnrollmentEnv  # noqa: E402
from src.solver import solve  # noqa: E402

ADVISOR = "advisor@univ.example"
AREAS = {  # requirement -> course prefix
    "systems": "SYS", "theory": "THY", "ai": "AIM", "data": "DAT",
}
TITLES = {
    "SYS": ["Operating Systems", "Distributed Storage", "Networked Systems", "Systems Security"],
    "THY": ["Algorithms", "Complexity Theory", "Cryptography", "Graph Theory"],
    "AIM": ["Machine Learning", "Probabilistic Models", "Planning", "Vision"],
    "DAT": ["Data Systems", "Stream Processing", "Data Mining", "Query Optimization"],
    "ELC": ["Technical Writing", "Ethics in Computing", "HCI Studio", "Startup Lab"],
}
SLOTS = [("MWF", 8 * 60, 50), ("MWF", 10 * 60, 50), ("MWF", 13 * 60, 50),
         ("MW", 14 * 60, 80), ("MW", 16 * 60, 80),
         ("TR", 9 * 60 + 30, 80), ("TR", 11 * 60, 80), ("TR", 14 * 60, 80)]
CORE = ["CORE 101", "CORE 120", "CORE 140"]
SPEC = {
    "easy":   {"n_req": 1, "drops": (0, 0), "depth": 1},
    "medium": {"n_req": 2, "drops": (0, 1), "depth": 2},
    "hard":   {"n_req": 3, "drops": (1, 1), "depth": 3},
}


def sec(slot, cap):
    days, start, dur = slot
    return {"days": days, "start": start, "end": start + dur, "capacity": cap}


def build_scenario(rng: random.Random, reqs: List[str], depth: int, force_drop: bool) -> Dict[str, Any]:
    catalog: Dict[str, Any] = {}
    for i, code in enumerate(CORE):
        catalog[code] = {"title": f"Core Computing {i + 1}", "units": 4,
                         "prereqs": [CORE[i - 1]] if i else [], "satisfies": ["core"],
                         "sections": {code.replace(" ", "") + "-A": sec(rng.choice(SLOTS), 40)}}
    completed = ["CORE 101", "CORE 120"]
    for area in AREAS:
        prefix = AREAS[area]
        titles = rng.sample(TITLES[prefix], 3)
        prev = None
        for j, title in enumerate(titles):
            num = 300 + 50 * j + rng.randrange(1, 49)
            code = f"{prefix} {num}"
            # chain: course j may require course j-1 (up to `depth`), else a core course
            if prev and j < depth and rng.random() < 0.6:
                prereqs = [prev]
            else:
                prereqs = [rng.choice(CORE[:3])]
            n_sec = rng.choice([1, 1, 2])
            sections = {}
            for k, slot in enumerate(rng.sample(SLOTS, n_sec)):
                sections[f"{prefix}{num}-{'AB'[k]}"] = sec(slot, rng.choice([2, 3, 3, 4]))
            catalog[code] = {"title": title, "units": 4, "prereqs": prereqs,
                             "satisfies": [area], "sections": sections}
            prev = code
        # sometimes the student already completed the first course in an area
        if rng.random() < 0.4:
            first_code = [c for c in catalog if c.startswith(prefix)][0]
            completed.append(first_code)
    # electives the student is currently enrolled in
    n_cur = 1 if not force_drop else rng.choice([1, 2])
    enrolled_secs, droppable = [], []
    for e in range(n_cur):
        code = f"ELC {110 + 7 * e + rng.randrange(0, 6)}"
        sid = code.replace(" ", "") + "-A"
        catalog[code] = {"title": TITLES["ELC"][e], "units": 4, "prereqs": [],
                         "satisfies": ["elective"], "sections": {sid: sec(rng.choice(SLOTS), 3)}}
        enrolled_secs.append(sid)
    droppable = [c for c in catalog if c.startswith("ELC")]
    n_needed = n_cur + len(reqs)
    max_units = 4 * n_needed - (4 if force_drop and rng.random() < 0.5 else 0)
    max_units = max(max_units, 8)
    # seat pressure: pre-fill sections with other students
    initial_enrolled: Dict[str, List[str]] = {}
    sid_n = 1
    for c in catalog.values():
        for sid, s in c["sections"].items():
            k = rng.randrange(0, s["capacity"])
            initial_enrolled[sid] = [f"S{sid_n + i}" for i in range(k)]
            sid_n += k
    # someone is waiting on each elective -> dropping it loses the seat for good
    initial_waitlists = {sid: [f"W{i + 1}"] for i, sid in enumerate(enrolled_secs)}
    for sid in enrolled_secs:
        cap = next(c["sections"][sid]["capacity"] for c in catalog.values() if sid in c["sections"])
        initial_enrolled[sid] = initial_enrolled[sid][:cap - 1]
    return {
        "catalog": catalog,
        "student": {"completed": sorted(set(completed)), "enrolled": enrolled_secs, "max_units": max_units},
        "initial_enrolled": initial_enrolled,
        "initial_waitlists": initial_waitlists,
        "competitors": [],
        "policy": {"allowed_recipients": [ADVISOR], "droppable": droppable},
    }


def add_competitors(rng: random.Random, scenario: Dict[str, Any], sol: Dict[str, Any]) -> None:
    """Competitors target the sections the correct plan needs, arriving after
    the clean run finishes: harmless to clean runs, harmful to slow recovery."""
    targets = list(sol["assign"].values())
    comps = []
    for i, sid in enumerate(targets):
        comps.append({"student": f"C{i + 1}", "section": sid, "arrive_at": 3 + rng.randrange(0, 3)})
    scenario["competitors"] = comps


def goal_text(reqs: List[str], droppable: List[str]) -> str:
    want = ", ".join(f"one {r} course" for r in reqs)
    drop = (f"I may drop {' or '.join(droppable)} if that is needed to fit them, but do not drop "
            f"anything unnecessarily." if droppable else "Do not drop any of my current classes.")
    return (f"Please enroll me in {want} for next term. {drop} "
            f"Afterwards email my advisor ({ADVISOR}) a short summary of the changes.")


def make_task(rng: random.Random, tid: str, difficulty: str) -> Optional[Dict[str, Any]]:
    spec = SPEC[difficulty]
    reqs = rng.sample(list(AREAS), spec["n_req"])
    scenario = build_scenario(rng, reqs, spec["depth"], force_drop=spec["drops"][0] > 0)
    env = CourseEnrollmentEnv(scenario)
    droppable = scenario["policy"]["droppable"]
    sol = solve(env.true_facts(), reqs, droppable=droppable)
    if sol is None:
        return None
    lo, hi = spec["drops"]
    if not lo <= len(sol["drops"]) <= hi:
        return None
    # the required courses must not be trivially the only option everywhere:
    # require >= 2 eligible-looking candidates per requirement (faults can redirect)
    facts = env.true_facts()
    for r in reqs:
        cands = [sid for c in facts["courses"].values() if r in c["satisfies"] for sid in
                 (s["section"] for s in c["sections"])]
        if len(cands) < 2:
            return None
    add_competitors(rng, scenario, sol)
    return {
        "task_id": tid, "difficulty": difficulty, "check": "enroll_goal",
        "goal": goal_text(reqs, droppable),
        "requirements": reqs,
        "constraints": {"droppable": droppable, "email_to": ADVISOR},
        "inputs": {"scenario": scenario},
        "expected": sol,
    }


def default_task() -> Dict[str, Any]:
    """t000: the hand-written worked example (SYS 455 / DATA 330)."""
    sc = copy.deepcopy(DEFAULT_SCENARIO)
    reqs = ["systems", "theory"]
    droppable = sc["policy"]["droppable"]
    sol = solve(CourseEnrollmentEnv(sc).true_facts(), reqs, droppable=droppable)
    return {"task_id": "t000", "difficulty": "example", "check": "enroll_goal",
            "goal": goal_text(reqs, droppable), "requirements": reqs,
            "constraints": {"droppable": droppable, "email_to": ADVISOR},
            "inputs": {"scenario": sc}, "expected": sol}


def generate(seed: int, counts: Dict[str, int], max_tries: int = 5000) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    tasks = [default_task()]
    n = 1
    for difficulty, k in counts.items():
        made, tries = 0, 0
        while made < k:
            tries += 1
            if tries > max_tries:
                raise RuntimeError(f"could not generate {k} {difficulty} tasks")
            t = make_task(rng, f"t{n:03d}", difficulty)
            if t is not None:
                tasks.append(t)
                made += 1
                n += 1
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--easy", type=int, default=6)
    ap.add_argument("--medium", type=int, default=8)
    ap.add_argument("--hard", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "data" / "tasks.json"))
    args = ap.parse_args()
    tasks = generate(args.seed, {"easy": args.easy, "medium": args.medium, "hard": args.hard})
    Path(args.out).write_text(json.dumps({"seed": args.seed, "tasks": tasks}, indent=1))
    by = {}
    for t in tasks:
        by[t["difficulty"]] = by.get(t["difficulty"], 0) + 1
    print(f"wrote {len(tasks)} tasks -> {args.out}  {by}")


if __name__ == "__main__":
    main()
