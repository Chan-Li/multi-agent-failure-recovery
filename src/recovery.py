"""Recovery strategies.

All strategies receive the identity of the faulty artifact (oracle), so
recovery quality is measured independently of detection.

Phase 1 baselines:
  - FullRestart: restore initial env state and rerun every stage.
  - VerifyAll:   verify each intermediate artifact before passing it on
                 (prevention-style baseline; costs extra inference).
Phase 2:
  - DependencyAwareInvalidation: invalidate only downstream dependents,
    restore affected state, rerun only necessary stages.

Every strategy must report irreversible actions that already happened
instead of counting recomputation as complete recovery.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .tracing import ToolAction
from .workflow import RunResult, Workflow


@dataclass
class RecoveryReport:
    method: str
    recovered_success: bool
    recomputed_stages: List[str]
    retained_artifacts: List[str]
    irreversible_actions: List[ToolAction] = field(default_factory=list)
    recovery_latency_s: float = 0.0
    extra_llm_tokens: int = 0
    tracking_overhead_s: float = 0.0
    final: RunResult = None


class RecoveryStrategy(ABC):
    name: str = "base"

    @abstractmethod
    def recover(self, workflow: Workflow, task: Dict[str, Any],
                faulty_run: RunResult, faulty_artifact_id: str) -> RecoveryReport:
        ...


class FullRestart(RecoveryStrategy):
    name = "full_restart"

    def recover(self, workflow, task, faulty_run, faulty_artifact_id):
        raise NotImplementedError


class VerifyAll(RecoveryStrategy):
    name = "verify_all"

    def recover(self, workflow, task, faulty_run, faulty_artifact_id):
        raise NotImplementedError


class DependencyAwareInvalidation(RecoveryStrategy):
    """Phase 2."""
    name = "dependency_aware"

    def recover(self, workflow, task, faulty_run, faulty_artifact_id):
        raise NotImplementedError


STRATEGIES = {s.name: s for s in (FullRestart, VerifyAll, DependencyAwareInvalidation)}
