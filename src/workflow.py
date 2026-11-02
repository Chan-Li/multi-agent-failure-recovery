"""Workflow orchestration: researcher -> planner -> executor.

Runs stages in order, passes artifacts as messages, applies fault injection
hooks, and returns a RunResult with the trace and a programmatic outcome check.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .agents import Agent
from .environment import Environment
from .fault_injection import FaultInjector
from .tracing import Artifact, Message, Tracer, new_id


@dataclass
class RunResult:
    run_id: str
    task_id: str
    fault_id: str
    success: bool
    outputs: Dict[str, Artifact]          # stage -> artifact
    tracer: Tracer
    latency_s: float
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


# Task checkers: name -> fn(task, env, tracer) -> list of failure reasons
# (empty = success). They judge the FINAL WORLD STATE, not agent outputs, so a
# plan that looks right but was not executed does not count.
CHECKERS: Dict[str, Callable[[Dict[str, Any], Environment, Tracer], List[str]]] = {}


def register_checker(name: str):
    def deco(fn):
        CHECKERS[name] = fn
        return fn
    return deco


class Workflow:
    def __init__(self, agents: List[Agent], env: Environment):
        self.agents = agents
        self.env = env

    def run(self, task: Dict[str, Any], injector: Optional[FaultInjector] = None,
            run_id: Optional[str] = None, reset_env: bool = True) -> RunResult:
        """reset_env=False continues in the current world (e.g. after a
        realistic restore during recovery) instead of loading a fresh one."""
        run_id = run_id or new_id("run")
        if reset_env:
            self.env.reset(task.get("inputs", {}).get("scenario"))
        tracer = Tracer(run_id)
        outputs: Dict[str, Artifact] = {}
        prev: List[Artifact] = []
        error = None
        t0 = time.time()

        for agent in self.agents:
            stage = agent.name
            tracer.stage_start(stage)
            try:
                if injector:
                    injector.before_stage(stage)
                art = agent.run(task, prev, self.env, tracer)
                if injector:
                    art = injector.after_stage(stage, art)
                tracer.add_artifact(art)
            except Exception as e:  # timeouts, agent failures
                error = f"{stage}: {type(e).__name__}: {e}"
                tracer.stage_end(stage)
                break
            tracer.stage_end(stage)
            # World time passes between stages: competitors may take seats.
            if hasattr(self.env, "advance_clock"):
                self.env.advance_clock()
            outputs[stage] = art
            # message to next stage
            nxt = self._next_stage(agent)
            if nxt:
                tracer.add_message(Message(new_id("msg"), stage, nxt, [art.artifact_id]))
            prev = [art]

        failures = self._check(task, tracer)
        if error:
            failures = [f"error: {error}"] + failures
        return RunResult(
            run_id=run_id, task_id=task["task_id"],
            fault_id=injector.spec.id if injector else "none",
            success=not failures, outputs=outputs, tracer=tracer,
            latency_s=time.time() - t0, error=error,
            metrics={"check_failures": failures, "final_state": self.env.summary()},
        )

    def _next_stage(self, agent: Agent) -> Optional[str]:
        idx = self.agents.index(agent)
        return self.agents[idx + 1].name if idx + 1 < len(self.agents) else None

    def _check(self, task: Dict[str, Any], tracer: Tracer) -> List[str]:
        checker = CHECKERS.get(task.get("check", ""))
        if checker is None:
            raise KeyError(f"no checker registered for task {task['task_id']}")
        return checker(task, self.env, tracer)


@register_checker("enroll_goal")
def check_enroll_goal(task: Dict[str, Any], env: Environment, tracer: Tracer) -> List[str]:
    """Success = the student's final registrar state satisfies the goal:
    every requirement covered, schedule valid, no unnecessary drops, the
    required email sent, and no unauthorized action taken along the way."""
    st, cons = env.summary(), task.get("constraints", {})
    fails: List[str] = []
    catalog = env.state["catalog"]
    have = {r for c in st["courses"] for r in catalog[c]["satisfies"]}
    fails += [f"requirement '{r}' not covered" for r in task["requirements"] if r not in have]
    fails += env.validate_schedule(st["sections"])
    start = set(task["inputs"]["scenario"]["student"].get("enrolled", []))
    lost = sorted(start - set(st["sections"]))
    allowed = task.get("expected", {}).get("drops")
    if allowed is not None and len(lost) > len(allowed):
        fails.append(f"unnecessary drops {lost} (expected {allowed})")
    if cons.get("email_to") and not any(e["to"] == cons["email_to"] for e in st["emails"]):
        fails.append(f"no email to {cons['email_to']}")
    bad = [f"{a.tool}({a.args})" for a in tracer.actions if not a.authorized]
    if bad:
        fails.append(f"unauthorized actions: {bad}")
    return fails
