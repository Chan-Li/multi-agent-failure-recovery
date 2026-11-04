"""Tracing: records messages, artifacts, tool actions, and dependency edges.

Every intermediate result is an Artifact with a unique id and a list of the
artifact ids it was derived from (`sources`). This provenance graph is what
Phase 1 uses to measure propagation and Phase 2 uses for invalidation.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@dataclass
class Artifact:
    """An intermediate result produced by an agent (or a tool)."""
    artifact_id: str
    producer: str                      # stage / agent name
    content: Any
    sources: List[str] = field(default_factory=list)   # upstream artifact ids
    state_reads: List[str] = field(default_factory=list)  # shared-state keys read
    faulty: bool = False               # ground truth: set by the injector
    fault_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)  # e.g. parse_failed, fallback
    created_at: float = field(default_factory=time.time)


@dataclass
class Message:
    message_id: str
    sender: str
    receiver: str
    artifact_ids: List[str]
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolAction:
    action_id: str
    stage: str
    tool: str
    args: Dict[str, Any]
    result: Any = None
    sources: List[str] = field(default_factory=list)   # artifacts that motivated the call
    state_writes: List[str] = field(default_factory=list)
    reversible: bool = True
    authorized: bool = True
    timestamp: float = field(default_factory=time.time)


@dataclass
class LLMCall:
    call_id: str
    stage: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    purpose: str = "work"              # work | verify | recovery
    finish_reason: Optional[str] = None  # API stop reason: "stop", "length" (budget hit), ...
    text: Optional[str] = None           # raw reply, so parse failures can be diagnosed
    attempt: int = 0                     # 0 = first try, 1 = retry after a parse failure
    max_tokens: Optional[int] = None     # output budget of this call


class Tracer:
    """Collects all events for one workflow run."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.artifacts: Dict[str, Artifact] = {}
        self.messages: List[Message] = []
        self.actions: List[ToolAction] = []
        self.llm_calls: List[LLMCall] = []
        self.stage_timings: Dict[str, Dict[str, float]] = {}

    # --- recording -------------------------------------------------------
    def add_artifact(self, artifact: Artifact) -> Artifact:
        self.artifacts[artifact.artifact_id] = artifact
        return artifact

    def add_message(self, msg: Message) -> None:
        self.messages.append(msg)

    def add_action(self, action: ToolAction) -> None:
        self.actions.append(action)

    def add_llm_call(self, call: LLMCall) -> None:
        self.llm_calls.append(call)

    def stage_start(self, stage: str) -> None:
        self.stage_timings.setdefault(stage, {})["start"] = time.time()

    def stage_end(self, stage: str) -> None:
        self.stage_timings.setdefault(stage, {})["end"] = time.time()

    # --- dependency queries ---------------------------------------------
    def downstream_of(self, artifact_id: str) -> Set[str]:
        """Transitive closure of artifacts derived from `artifact_id`."""
        children: Dict[str, List[str]] = {}
        for a in self.artifacts.values():
            for s in a.sources:
                children.setdefault(s, []).append(a.artifact_id)
        seen: Set[str] = set()
        stack = [artifact_id]
        while stack:
            cur = stack.pop()
            for c in children.get(cur, []):
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return seen

    def actions_depending_on(self, artifact_ids: Set[str]) -> List[ToolAction]:
        return [a for a in self.actions if set(a.sources) & artifact_ids]

    # --- persistence -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "artifacts": [asdict(a) for a in self.artifacts.values()],
            "messages": [asdict(m) for m in self.messages],
            "actions": [asdict(a) for a in self.actions],
            "llm_calls": [asdict(c) for c in self.llm_calls],
            "stage_timings": self.stage_timings,
        }

    def save(self, directory: str | Path) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"{self.run_id}.json"
        out.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return out
