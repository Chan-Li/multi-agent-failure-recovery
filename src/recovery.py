"""Recovery strategies.

All strategies receive the identity of the faulty artifact (oracle), so
recovery quality is measured independently of detection.

Phase 1 baselines:
  - FullRestart: restore the start-of-run snapshot and rerun every stage.
      realistic restore: our drops / sent mail / competitor arrivals persist
                         and the clock keeps running (the real world)
      oracle restore:    everything rolls back (unreachable upper bound)
  - VerifyAll:   prevention baseline -- the same task under the same fault
                 schedule, but every intermediate artifact is verified before
                 it is passed on (extra inference + registrar reads).
Phase 2:
  - DependencyAwareInvalidation: invalidate only downstream dependents,
    restore affected state, rerun only necessary stages.

Every strategy reports irreversible actions that already happened instead of
counting recomputation as complete recovery.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .agents import Agent
from .fault_injection import FaultInjector, FaultSpec
from .tracing import ToolAction
from .workflow import RunResult, Workflow


@dataclass
class RecoveryReport:
    method: str
    recovered_success: bool                 # goal met in the final world state
    recomputed_stages: List[str]
    retained_artifacts: List[str]
    irreversible_actions: List[ToolAction] = field(default_factory=list)   # fault-caused, already done
    harmful_irreversible: List[ToolAction] = field(default_factory=list)   # unauthorized / unneeded drops
    persisted_events: int = 0               # world events a restore could not undo
    recovery_latency_s: float = 0.0
    extra_llm_tokens: int = 0
    tracking_overhead_s: float = 0.0
    detected: Optional[bool] = None         # verify_all: was the faulty artifact rejected?
    false_alarms: int = 0                   # verify_all: clean artifacts rejected
    failures: List[str] = field(default_factory=list)
    final: Optional[RunResult] = None
    caught_downstream: Optional[bool] = None   # verify_all: fault missed, its effect caught later
    verifier_parse_failures: int = 0           # verify_all: verdicts still unparseable after retries
    verifier_retries: int = 0                  # verify_all: verdicts that needed a retry to parse

    @property
    def clean_recovery(self) -> bool:
        """Goal met AND no harmful side effect of the fault survives."""
        return self.recovered_success and not self.harmful_irreversible

    def to_row(self, prefix: str) -> Dict[str, Any]:
        return {
            f"{prefix}.success": self.recovered_success,
            f"{prefix}.clean_recovery": self.clean_recovery,
            f"{prefix}.recomputed": self.recomputed_stages,
            f"{prefix}.n_recomputed": len(self.recomputed_stages),
            f"{prefix}.retained": len(self.retained_artifacts),
            f"{prefix}.irreversible": len(self.irreversible_actions),
            f"{prefix}.harmful_irreversible": len(self.harmful_irreversible),
            f"{prefix}.persisted_events": self.persisted_events,
            f"{prefix}.latency_s": self.recovery_latency_s,
            f"{prefix}.extra_tokens": self.extra_llm_tokens,
            f"{prefix}.detected": self.detected,
            f"{prefix}.false_alarms": self.false_alarms,
            f"{prefix}.failures": self.failures,
            f"{prefix}.caught_downstream": self.caught_downstream,
            f"{prefix}.verifier_parse_failures": self.verifier_parse_failures,
            f"{prefix}.verifier_retries": self.verifier_retries,
        }


def tokens(run: RunResult, purpose: Optional[str] = None) -> int:
    return sum(c.prompt_tokens + c.completion_tokens for c in run.tracer.llm_calls
               if purpose is None or c.purpose == purpose)


def regeneration_tokens(run: RunResult) -> int:
    """Work tokens beyond the first LLM call of each stage (= regenerations)."""
    seen, extra = set(), 0
    for c in run.tracer.llm_calls:
        if c.purpose != "work":
            continue
        if c.stage in seen:
            extra += c.prompt_tokens + c.completion_tokens
        seen.add(c.stage)
    return extra


def harmful_subset(irreversible: List[Any], n_drops_needed: int,
                   get=lambda a, k: getattr(a, k)) -> List[Any]:
    """Harmful irreversible actions: every unauthorized one, plus authorized
    drops BEYOND the number the goal needs. Counting (not matching section
    ids) matters: when two electives are equally droppable, dropping the
    other one is an equally correct plan, not harm. Works on ToolAction
    objects or, with `get`, on trace dicts."""
    harmful = [a for a in irreversible if not get(a, "authorized")]
    drops = [a for a in irreversible if get(a, "tool") == "drop" and get(a, "authorized")]
    harmful += drops[n_drops_needed:]
    return harmful


def fault_side_effects(run: RunResult, task: Dict[str, Any], faulty_artifact_id: Optional[str]):
    """Irreversible actions in `run` that depend on the faulty artifact, and the
    harmful subset (see harmful_subset)."""
    tr = run.tracer
    if faulty_artifact_id is None or faulty_artifact_id not in tr.artifacts:
        return [], []
    tainted = {faulty_artifact_id} | tr.downstream_of(faulty_artifact_id)
    irrev = [a for a in tr.actions_depending_on(tainted) if not a.reversible]
    needed = len(task.get("expected", {}).get("drops", []))
    return irrev, harmful_subset(irrev, needed)


class RecoveryStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def recover(self, workflow: Workflow, task: Dict[str, Any], faulty_run: RunResult,
                faulty_artifact_id: Optional[str], spec: Optional[FaultSpec] = None) -> RecoveryReport:
        """`workflow.env` must be the world as the faulty run left it."""


class FullRestart(RecoveryStrategy):
    name = "full_restart"

    def __init__(self, restore_mode: str = "realistic"):
        self.restore_mode = restore_mode

    def recover(self, workflow, task, faulty_run, faulty_artifact_id, spec=None):
        irrev, harmful = fault_side_effects(faulty_run, task, faulty_artifact_id)
        t0 = time.time()
        persisted = workflow.env.restore(faulty_run.metrics["start_snapshot"], self.restore_mode)
        rerun = workflow.run(task, injector=None, reset_env=False,
                             run_id=f"{faulty_run.run_id}-{self.name}-{self.restore_mode}")
        failures = list(rerun.metrics["check_failures"])
        # the rerun's checker only sees the rerun; the faulty run's harm is still out there
        if self.restore_mode == "oracle":
            harmful = []                     # oracle rolls the world back completely
        return RecoveryReport(
            method=f"{self.name}_{self.restore_mode}" if self.restore_mode != "realistic" else self.name,
            recovered_success=rerun.success,
            recomputed_stages=[a.name for a in workflow.agents],
            retained_artifacts=[],
            irreversible_actions=irrev, harmful_irreversible=harmful,
            persisted_events=len(persisted),
            recovery_latency_s=time.time() - t0,
            extra_llm_tokens=tokens(rerun),
            failures=failures, final=rerun,
        )


class VerifyAll(RecoveryStrategy):
    """Not a recovery *after* the fact: re-executes the task from the same
    initial world under the same fault schedule, verifying every intermediate
    artifact. Compared on the same task x fault pairs as FullRestart."""
    name = "verify_all"

    def __init__(self, verifier: Agent, label: str = "verify_all"):
        self.verifier = verifier
        self.label = label          # distinguishes variants in run ids / trace files

    def recover(self, workflow, task, faulty_run, faulty_artifact_id, spec=None):
        t0 = time.time()
        injector = FaultInjector(spec, task) if spec is not None else None
        run = workflow.run(task, injector=injector, reset_env=True, verifier=self.verifier,
                           run_id=f"{faulty_run.run_id}-{self.label}")
        verdicts = run.metrics["verdicts"]
        detected = any(v["artifact_faulty"] and not v["ok"] for v in verdicts) if injector else None
        # a rejection of a clean artifact built on a faulty one is a downstream
        # catch, not a false alarm
        false_alarms = sum(1 for v in verdicts if not v["artifact_faulty"]
                           and not v.get("upstream_faulty") and not v["ok"])
        caught_downstream = any(v.get("upstream_faulty") and not v["ok"] for v in verdicts)
        parse_failures = sum(1 for v in verdicts if v.get("parse_failed"))
        retries = sum(1 for v in verdicts if v.get("retried"))
        faulty = next((a.artifact_id for a in run.tracer.artifacts.values() if a.faulty), None)
        irrev, harmful = fault_side_effects(run, task, faulty)
        return RecoveryReport(
            method=self.label, recovered_success=run.success,
            recomputed_stages=list(run.metrics["regenerated"]),
            retained_artifacts=[a for a in run.tracer.artifacts if not run.tracer.artifacts[a].faulty],
            irreversible_actions=irrev, harmful_irreversible=harmful,
            recovery_latency_s=time.time() - t0,
            # overhead vs. an unverified run: verifier calls + regenerations
            extra_llm_tokens=tokens(run, "verify") + regeneration_tokens(run),
            detected=detected, false_alarms=false_alarms,
            failures=list(run.metrics["check_failures"]), final=run,
            caught_downstream=caught_downstream, verifier_parse_failures=parse_failures,
            verifier_retries=retries,
        )


class DependencyAwareInvalidation(RecoveryStrategy):
    """Phase 2."""
    name = "dependency_aware"

    def recover(self, workflow, task, faulty_run, faulty_artifact_id, spec=None):
        raise NotImplementedError


STRATEGIES = {s.name: s for s in (FullRestart, VerifyAll, DependencyAwareInvalidation)}
