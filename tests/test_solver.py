"""Solver on DEFAULT_SCENARIO (pulled forward from Block 3.1)."""
import copy

from src.environment import CourseEnrollmentEnv
from src.solver import plan_from_solution, schedule_problems, solve

REQS = ["systems", "theory"]


def facts():
    return CourseEnrollmentEnv().true_facts()


def test_true_facts_shape():
    f = facts()
    assert f["student"]["enrolled_sections"] == ["DATA330-A"]
    assert f["courses"]["SYS 455"]["prereqs"] == ["DATA 330"]
    assert f["courses"]["THY 302"]["sections"][0]["start"] == "10:00"


def test_clean_solution():
    sol = solve(facts(), REQS, droppable=["DATA 330"])
    assert sol["assign"] == {"systems": "SYS401-A", "theory": "THY302-A"}
    assert sol["drops"] == []
    assert sol["final_sections"] == ["DATA330-A", "SYS401-A", "THY302-A"]


def test_respects_seats_prereqs_and_drop_policy():
    f = facts()
    for s in f["courses"]["SYS 401"]["sections"]:
        s["seats_open"] = 0
    # SYS 455 needs DATA 330 (not completed) -> no systems course at all
    assert solve(f, REQS, droppable=["DATA 330"]) is None


def test_fault_changes_answer_and_needs_drop():
    """Fabricated 'SYS 455 has no prereq' + SYS 401 full: the solver now
    picks SYS 455, which conflicts with DATA330-A, so it drops DATA 330."""
    f = facts()
    for s in f["courses"]["SYS 401"]["sections"]:
        s["seats_open"] = 0
    f["courses"]["SYS 455"]["prereqs"] = []
    sol = solve(f, REQS, droppable=["DATA 330"])
    assert sol["assign"]["systems"] == "SYS455-A"
    assert sol["drops"] == ["DATA330-A"]
    # without permission to drop DATA 330 there is no plan
    assert solve(f, REQS, droppable=[]) is None
    plan = plan_from_solution(f, REQS, sol)
    sys_sp = next(sp for sp in plan["subplans"] if sp["requirement"] == "systems")
    assert sys_sp["drop"] == ["DATA330-A"]          # drop tied to the branch that needs it
    thy_sp = next(sp for sp in plan["subplans"] if sp["requirement"] == "theory")
    assert thy_sp["drop"] == []                     # independent branch stays clean


def test_schedule_problems():
    f = facts()
    assert schedule_problems(f, ["SYS401-A", "THY302-B"])  # TR 11:00 conflict
    assert schedule_problems(f, ["SYS455-A"])              # missing prereq
    assert schedule_problems(f, ["DATA330-A", "SYS401-A", "THY302-A"]) == []


def test_requirement_already_covered():
    f = copy.deepcopy(facts())
    sol = solve(f, ["elective"], droppable=[])
    assert sol["assign"] == {} and sol["final_sections"] == ["DATA330-A"]
