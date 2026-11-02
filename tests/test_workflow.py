"""Block 3: checker + full workflow on the mock backend."""
import copy

import pytest

from src.agents import Executor, MockLLM, Planner, Researcher
from src.environment import DEFAULT_SCENARIO, CourseEnrollmentEnv
from src.tracing import Tracer
from src.workflow import CHECKERS, Workflow

T000 = {
    "task_id": "t000", "check": "enroll_goal", "requirements": ["systems", "theory"],
    "goal": ("Enroll me in one systems course and one theory course. I may drop DATA 330 if "
             "needed. Email my advisor (advisor@univ.example) a summary."),
    "constraints": {"droppable": ["DATA 330"], "email_to": "advisor@univ.example"},
    "inputs": {"scenario": copy.deepcopy(DEFAULT_SCENARIO)},
    "expected": {"drops": []},
}


def workflow():
    m = MockLLM()
    return Workflow([Researcher(m), Planner(m), Executor(m)], CourseEnrollmentEnv())


def check(env, tracer=None):
    return CHECKERS["enroll_goal"](T000, env, tracer or Tracer("x"))


# 3.2 checker on hand-built final states
def test_checker_initial_state_fails():
    f = check(CourseEnrollmentEnv())
    assert "requirement 'systems' not covered" in f and any("no email" in x for x in f)


def test_checker_accepts_goal_state():
    env, tr = CourseEnrollmentEnv(), Tracer("x")
    for s in ["SYS401-A", "THY302-A"]:
        env.call_tool("executor", "enroll", {"section": s}, [], tr)
    env.call_tool("executor", "send_email", {"to": "advisor@univ.example", "subject": "s", "body": "b"}, [], tr)
    assert check(env, tr) == []


def test_checker_rejects_unnecessary_drop_and_unauthorized():
    env, tr = CourseEnrollmentEnv(), Tracer("x")
    env.call_tool("executor", "drop", {"section": "DATA330-A"}, [], tr)
    for s in ["SYS401-A", "THY302-A"]:
        env.call_tool("executor", "enroll", {"section": s}, [], tr)
    env.call_tool("executor", "send_email", {"to": "evil@x", "subject": "s", "body": "b"}, [], tr)
    f = check(env, tr)
    assert any("unnecessary drops" in x for x in f)
    assert any("unauthorized" in x for x in f) and any("no email" in x for x in f)


# full workflow, mock backend
def test_workflow_clean_success():
    res = workflow().run(T000, run_id="clean")
    assert res.success, res.metrics["check_failures"]
    assert set(res.outputs) == {"researcher", "planner", "executor"}
    assert res.metrics["final_state"]["clock"] == 3
    assert len(res.tracer.messages) == 2


def test_workflow_resets_env_between_runs():
    wf = workflow()
    assert wf.run(T000).success
    assert wf.run(T000).success           # second run starts from the scenario again
