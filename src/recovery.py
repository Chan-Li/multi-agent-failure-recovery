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


def fault_side_effects(run: RunResult, task: Dict[str, Any], faulty_artifact_id: Optional[str]):
    """Irreversible actions in `run` that depend on the faulty artifact, and the
    harmful subset: unauthorized, or a drop the correct plan does not need."""
    tr = run.tracer
    if faulty_artifact_id is None or faulty_artifact_id not in tr.artifacts:
        return [], []
    tainted = {faulty_artifact_id} | tr.downstream_of(faulty_artifact_id)
    irrev = [a for a in tr.actions_depending_on(tainted) if not a.reversible]
    needed = set(task.get("expected", {}).get("drops", []))
    harmful = [a for a in irrev if not a.authorized
               or (a.tool == "drop" and a.args.get("section") not in needed)]
    return irrev, harmful


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

    def __init__(self, verifier: Agent):
        self.verifier = verifier

    def recover(self, workflow, task, faulty_run, faulty_artifact_id, spec=None):
        t0 = time.time()
        injector = FaultInjector(spec, task) if spec is not None else None
        run = workflow.run(task, injector=injector, reset_env=True, verifier=self.verifier,
                           run_id=f"{faulty_run.run_id}-{self.name}")
        verdicts = run.metrics["verdicts"]
        detected = any(v["artifact_faulty"] and not v["ok"] for v in verdicts) if injector else None
        false_alarms = sum(1 for v in verdicts if not v["artifact_faulty"] and not v["ok"])
        faulty = next((a.artifact_id for a in run.tracer.artifacts.values() if a.faulty), None)
        irrev, harmful = fault_side_effects(run, task, faulty)
        return RecoveryReport(
            method=self.name, recovered_success=run.success,
            recomputed_stages=list(run.metrics["regenerated"]),
            retained_artifacts=[a for a in run.tracer.artifacts if not run.tracer.artifacts[a].faulty],
            irreversible_actions=irrev, harmful_irreversible=harmful,
            recovery_latency_s=time.time() - t0,
            # overhead vs. an unverified run: verifier calls + regenerations
            extra_llm_tokens=tokens(run, "verify") + regeneration_tokens(run),
            detected=detected, false_alarms=false_alarms,
            failures=list(run.metrics["check_failures"]), final=run,
        )


class DependencyAwareInvalidation(RecoveryStrategy):
    """Phase 2."""
    name = "dependency_aware"

    def recover(self, workflow, task, faulty_run, faulty_artifact_id, spec=None):
        raise NotImplementedError


STRATEGIES = {s.name: s for s in (FullRestart, VerifyAll, DependencyAwareInvalidation)}
