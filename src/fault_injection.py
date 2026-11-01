"""Fault injection harness.

A FaultSpec describes *what* to inject and *where*. The injector wraps a
stage's output artifact (post-hoc corruption) or the stage call itself
(timeouts). Injected artifacts are marked `faulty=True` as ground truth so
propagation can be measured; agents never see this flag.

Accidental faults (incorrect_result, timeout, stale_output) and adversarial
faults (malicious_instruction) are tagged by `category` and analyzed separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .tracing import Artifact

FAULT_KINDS = ("none", "incorrect_result", "malicious_instruction", "timeout", "stale_output")
CATEGORIES = ("accidental", "adversarial")


@dataclass
class FaultSpec:
    id: str
    kind: str = "none"
    category: Optional[str] = None
    stage: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.kind in FAULT_KINDS, f"unknown fault kind {self.kind}"
        if self.kind != "none":
            assert self.category in CATEGORIES, f"fault {self.id} needs a category"


class FaultInjector:
    def __init__(self, spec: FaultSpec):
        self.spec = spec
        self.fired = False

    def applies_to(self, stage: str) -> bool:
        return self.spec.kind != "none" and self.spec.stage == stage

    def before_stage(self, stage: str) -> None:
        """Hook for faults that prevent normal execution (e.g. timeout)."""
        if self.applies_to(stage) and self.spec.kind == "timeout":
            self.fired = True
            raise TimeoutError(f"injected timeout at {stage}")

    def after_stage(self, stage: str, artifact: Artifact) -> Artifact:
        """Corrupt the stage output according to the fault kind."""
        if not self.applies_to(stage) or self.spec.kind == "timeout":
            return artifact
        handler = {
            "incorrect_result": self._incorrect_result,
            "malicious_instruction": self._malicious_instruction,
            "stale_output": self._stale_output,
        }[self.spec.kind]
        artifact.content = handler(artifact.content)
        artifact.faulty = True
        artifact.fault_id = self.spec.id
        self.fired = True
        return artifact

    # --- domain-specific corruptions (TODO once domain is chosen) --------
    def _incorrect_result(self, content: Any) -> Any:
        raise NotImplementedError

    def _malicious_instruction(self, content: Any) -> Any:
        raise NotImplementedError

    def _stale_output(self, content: Any) -> Any:
        raise NotImplementedError


def load_fault_specs(cfg_faults: List[Dict[str, Any]]) -> List[FaultSpec]:
    return [FaultSpec(**f) for f in cfg_faults]
