"""Block 3: generated tasks are valid and the mock workflow solves every one (gate)."""
import json
import sys
from pathlib import Path

import pytest

from src.agents import Executor, MockLLM, Planner, Researcher
from src.environment import CourseEnrollmentEnv
from src.solver import solve
from src.workflow import Workflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from gen_tasks import generate  # noqa: E402

TASKS = json.loads((ROOT / "data" / "tasks.json").read_text())["tasks"]


def test_counts_and_ids():
    by = {}
    for t in TASKS:
        by[t["difficulty"]] = by.get(t["difficulty"], 0) + 1
    assert by == {"example": 1, "easy": 6, "medium": 8, "hard": 6}
    assert len({t["task_id"] for t in TASKS}) == len(TASKS)


def test_generator_is_deterministic():
    assert generate(0, {"easy": 2, "medium": 1, "hard": 1}) == generate(0, {"easy": 2, "medium": 1, "hard": 1})


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["task_id"])
def test_task_valid(task):
    sc = task["inputs"]["scenario"]
    env = CourseEnrollmentEnv(sc)
    sol = solve(env.true_facts(), task["requirements"], task["constraints"]["droppable"])
    assert sol == task["expected"]
    assert env.validate_schedule(sol["final_sections"]) == []
    assert all(c["arrive_at"] >= 2 for c in sc["competitors"])
    if task["difficulty"] == "hard":
        assert len(sol["drops"]) == 1
    if task["difficulty"] == "easy":
        assert sol["drops"] == []
    # each dropped section has someone waiting -> the drop is irreversible
    for d in sol["drops"]:
        assert sc["initial_waitlists"].get(d)


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t["task_id"])
def test_mock_clean_success(task):
    m = MockLLM()
    wf = Workflow([Researcher(m), Planner(m), Executor(m)], CourseEnrollmentEnv())
    res = wf.run(task)
    assert res.success, res.metrics["check_failures"]
    assert sorted(res.metrics["final_state"]["sections"]) == task["expected"]["final_sections"]
