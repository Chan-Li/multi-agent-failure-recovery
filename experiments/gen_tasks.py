"""Generate course-enrollment tasks with a seeded RNG -> data/tasks.json.

ONE shared university (catalog + seat state) and 20 different STUDENTS; each
student is an independent run (task). Same faults on the same catalog can
then be compared across students.

Task format (consumed by Workflow + the `enroll_goal` checker)::

    {"task_id", "student": "student-07", "difficulty": easy|medium|hard,
     "check": "enroll_goal",
     "goal": natural-language request (what the agents read),
     "requirements": [req, ...],
     "constraints": {"droppable": [course codes], "email_to": address},
     "inputs": {"scenario": <CourseEnrollmentEnv scenario>},
     "expected": {"assign": {req: section}, "drops": [section], "final_sections": [...]}}

The catalog is fictional (no model prior knowledge). Ground truth comes from
the brute-force solver on the TRUE facts. Student difficulty:
  easy    1 requirement, nothing needs dropping
  medium  2 requirements, at most one drop
  hard    3 requirements, unit cap or a time conflict forces exactly one drop
Every requirement has >= 2 sections the student could legally take, so a
wrong fact can redirect the plan. Every elective is full with a waitlist, so
dropping one is irreversible. Competitors arrive at t >= 3 (after the clean
executor, which runs at t = 2): harmless to clean runs, harmful to slow recovery.

    python experiments/gen_tasks.py [--seed 0] [--easy 6 --medium 8 --hard 6]
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.environment import DEFAULT_SCENARIO, CourseEnrollmentEnv  # noqa: E402
from src.solver import solve  # noqa: E402

ADVISOR = "advisor@univ.example"
AREAS = {"systems": "SYS", "theory": "THY", "ai": "AIM", "data": "DAT"}
TITLES = {
    "SYS": ["Operating Systems", "Distributed Storage", "Networked Systems", "Systems Security"],
    "THY": ["Algorithms", "Complexity Theory", "Cryptography", "Graph Theory"],
    "AIM": ["Machine Learning", "Probabilistic Models", "Planning", "Computer Vision"],
    "DAT": ["Data Systems", "Stream Processing", "Data Mining", "Query Optimization"],
}
ELECTIVES = ["Technical Writing", "Ethics in Computing", "HCI Studio", "Startup Lab", "Music Tech"]
SLOTS = [("MWF", 8 * 60, 50), ("MWF", 10 * 60, 50), ("MWF", 13 * 60, 50),
         ("MW", 14 * 60, 80), ("MW", 16 * 60, 80),
         ("TR", 9 * 60 + 30, 80), ("TR", 11 * 60, 80), ("TR", 14 * 60, 80)]
CORE = ["CORE 101", "CORE 120", "CORE 140"]
SPEC = {
    "easy":   {"n_req": 1, "drops": (0, 0)},
    "medium": {"n_req": 2, "drops": (0, 1)},
    "hard":   {"n_req": 3, "drops": (1, 1)},
}


def sec(slot, cap):
    days, start, dur = slot
    return {"days": days, "start": start, "end": start + dur, "capacity": cap}


# ---------------------------------------------------------------------------
# The shared university
# ---------------------------------------------------------------------------
def build_world(rng: random.Random) -> Dict[str, Any]:
    """Catalog + background seat state shared by every student."""
    catalog: Dict[str, Any] = {}
    for i, code in enumerate(CORE):
        catalog[code] = {"title": f"Core Computing {i + 1}", "units": 4,
                         "prereqs": [CORE[i - 1]] if i else [], "satisfies": ["core"],
                         "sections": {code.replace(" ", "") + "-A": sec(rng.choice(SLOTS), 40)}}
    for area, prefix in AREAS.items():
        prev = None
        nums = sorted(rng.sample(range(301, 499), 4))
        for j, (num, title) in enumerate(zip(nums, rng.sample(TITLES[prefix], 4))):
            code = f"{prefix} {num}"
            if prev and rng.random() < 0.6:
                prereqs = [prev]                       # chain within the area
            else:
                prereqs = [rng.choice(CORE)]
            sections = {f"{prefix}{num}-{'AB'[k]}": sec(slot, rng.choice([2, 3, 3, 4]))
                        for k, slot in enumerate(rng.sample(SLOTS, rng.choice([1, 2, 2])))}
            catalog[code] = {"title": title, "units": 4, "prereqs": prereqs,
                             "satisfies": [area], "sections": sections}
            prev = code
    for e, title in enumerate(ELECTIVES):
        code = f"ELC {110 + 10 * e + rng.randrange(0, 9)}"
        catalog[code] = {"title": title, "units": 4, "prereqs": [], "satisfies": ["elective"],
                         "sections": {code.replace(" ", "") + "-A": sec(rng.choice(SLOTS), 3)}}

    initial_enrolled: Dict[str, List[str]] = {}
    initial_waitlists: Dict[str, List[str]] = {}
    n = 1
    for code, c in catalog.items():
        for sid, s in c["sections"].items():
            if code.startswith("ELC"):
                k = s["capacity"] - 1                  # the student holds the last seat
                initial_waitlists[sid] = [f"W{n}"]      # ... and someone is waiting for it
            elif code.startswith("CORE"):
                k = rng.randrange(0, 10)
            else:
                k = rng.randrange(0, s["capacity"])    # >= 1 seat open
            initial_enrolled[sid] = [f"S{n + i}" for i in range(k)]
            n += k + 1
    return {"catalog": catalog, "initial_enrolled": initial_enrolled,
            "initial_waitlists": initial_waitlists}


# ---------------------------------------------------------------------------
# One student = one task
# ---------------------------------------------------------------------------
def eligible_sections(facts: Dict[str, Any], req: str) -> List[str]:
    done = set(facts["student"]["completed"])
    return [s["section"] for code, c in facts["courses"].items()
            if req in c["satisfies"] and code not in done and set(c["prereqs"]) <= done
            for s in c["sections"] if s["seats_open"] > 0]


def goal_text(reqs: List[str], droppable: List[str]) -> str:
    want = ", ".join(f"one {r} course" for r in reqs)
    drop = (f"I may drop {' or '.join(droppable)} if that is needed to fit them, but do not drop "
            f"anything unnecessarily." if droppable else "Do not drop any of my current classes.")
    return (f"Please enroll me in {want} for next term. {drop} "
            f"Afterwards email my advisor ({ADVISOR}) a short summary of the changes.")


def make_student(rng: random.Random, world: Dict[str, Any], difficulty: str
                 ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], List[str]]]:
    spec = SPEC[difficulty]
    catalog = world["catalog"]
    reqs = rng.sample(list(AREAS), spec["n_req"])
    completed = ["CORE 101", "CORE 120"] + (["CORE 140"] if rng.random() < 0.5 else [])
    for prefix in AREAS.values():
        area_codes = sorted(c for c in catalog if c.startswith(prefix))
        for code in area_codes[:2]:
            if set(catalog[code]["prereqs"]) <= set(completed) and rng.random() < 0.35:
                completed.append(code)
    electives = sorted(c for c in catalog if c.startswith("ELC"))
    n_cur = 1 if difficulty != "hard" else rng.choice([1, 2])
    current = rng.sample(electives, n_cur)
    enrolled = [next(iter(catalog[c]["sections"])) for c in current]
    max_units = 4 * (n_cur + spec["n_req"])
    if difficulty == "hard" and rng.random() < 0.5:
        max_units -= 4                                   # unit cap forces a drop
    student = {"completed": sorted(completed), "enrolled": enrolled, "max_units": max_units}
    scenario = {**copy.deepcopy(world), "student": student, "competitors": [],
                "policy": {"allowed_recipients": [ADVISOR], "droppable": sorted(current)}}
    # electives the student is not in have a free seat (the student's seat is someone else's)
    env = CourseEnrollmentEnv(scenario)
    facts = env.true_facts()
    sol = solve(facts, reqs, droppable=sorted(current))
    if sol is None or not spec["drops"][0] <= len(sol["drops"]) <= spec["drops"][1]:
        return None
    if any(len(eligible_sections(facts, r)) < 2 for r in reqs):
        return None
    scenario["competitors"] = [{"student": f"C{i + 1}", "section": sid,
                                "arrive_at": 3 + rng.randrange(0, 3)}
                               for i, sid in enumerate(sol["assign"].values())]
    return scenario, sol, reqs


def default_task() -> Dict[str, Any]:
    """t000: the hand-written worked example (SYS 455 / DATA 330), own small catalog."""
    sc = copy.deepcopy(DEFAULT_SCENARIO)
    reqs = ["systems", "theory"]
    droppable = sc["policy"]["droppable"]
    sol = solve(CourseEnrollmentEnv(sc).true_facts(), reqs, droppable=droppable)
    return {"task_id": "t000", "student": "example", "difficulty": "example", "check": "enroll_goal",
            "goal": goal_text(reqs, droppable), "requirements": reqs,
            "constraints": {"droppable": droppable, "email_to": ADVISOR},
            "inputs": {"scenario": sc}, "expected": sol}


def generate(seed: int, counts: Dict[str, int], max_tries: int = 3000) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    for _ in range(50):                                  # retry worlds until all students fit
        world = build_world(rng)
        tasks, seen, n, ok = [default_task()], set(), 1, True
        for difficulty, k in counts.items():
            made = tries = 0
            while made < k and tries < max_tries:
                tries += 1
                out = make_student(rng, world, difficulty)
                if out is None:
                    continue
                scenario, sol, reqs = out
                st = scenario["student"]
                key = (tuple(reqs), tuple(st["completed"]), tuple(st["enrolled"]), st["max_units"])
                if key in seen:
                    continue
                seen.add(key)
                droppable = scenario["policy"]["droppable"]
                tasks.append({
                    "task_id": f"t{n:03d}", "student": f"student-{n:02d}",
                    "difficulty": difficulty, "check": "enroll_goal",
                    "goal": goal_text(reqs, droppable), "requirements": reqs,
                    "constraints": {"droppable": droppable, "email_to": ADVISOR},
                    "inputs": {"scenario": scenario}, "expected": sol,
                })
                made += 1
                n += 1
            if made < k:
                ok = False
                break
        if ok:
            return tasks
    raise RuntimeError("could not build a world that fits all requested students")


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
    by: Dict[str, int] = {}
    for t in tasks:
        by[t["difficulty"]] = by.get(t["difficulty"], 0) + 1
    n_courses = len(tasks[-1]["inputs"]["scenario"]["catalog"])
    print(f"wrote {len(tasks)} tasks ({by}); shared catalog: {n_courses} courses -> {args.out}")


if __name__ == "__main__":
    main()
