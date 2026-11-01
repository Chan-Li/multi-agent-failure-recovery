"""Simulated environment: shared state + tools with side effects.

The environment is the "world" that agents act on. It must support
snapshot/restore so recovery can roll back simulated state, and it must
mark which tool actions are irreversible (e.g. sending an email, a payment)
so they are reported rather than silently "undone".
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .tracing import ToolAction, Tracer, new_id


@dataclass
class ToolSpec:
    name: str
    fn: Callable[..., Any]
    reversible: bool = True
    requires_authorization: bool = False   # used to flag unauthorized actions
    writes: Optional[List[str]] = None      # shared-state keys this tool may modify


class Environment:
    """Key-value shared state plus a tool registry. Domain-specific subclasses
    populate initial state and register tools."""

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        self.state: Dict[str, Any] = copy.deepcopy(initial_state or {})
        self.tools: Dict[str, ToolSpec] = {}
        self.authorized_tools: set = set()
        self._snapshots: Dict[str, Dict[str, Any]] = {}

    # --- tools -----------------------------------------------------------
    def register_tool(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def call_tool(self, stage: str, tool: str, args: Dict[str, Any],
                  sources: List[str], tracer: Tracer) -> Any:
        spec = self.tools[tool]
        authorized = (not spec.requires_authorization) or tool in self.authorized_tools
        result = spec.fn(self, **args)
        tracer.add_action(ToolAction(
            action_id=new_id("act"), stage=stage, tool=tool, args=args,
            result=result, sources=sources, state_writes=list(spec.writes or []),
            reversible=spec.reversible, authorized=authorized,
        ))
        return result

    # --- shared state ----------------------------------------------------
    def read(self, key: str) -> Any:
        return copy.deepcopy(self.state.get(key))

    def write(self, key: str, value: Any) -> None:
        self.state[key] = value

    # --- snapshots (in-process; swap for container/VM snapshots later) ---
    def snapshot(self, label: Optional[str] = None) -> str:
        sid = label or new_id("snap")
        self._snapshots[sid] = copy.deepcopy(self.state)
        return sid

    def restore(self, snapshot_id: str) -> None:
        self.state = copy.deepcopy(self._snapshots[snapshot_id])

    def reset(self) -> None:
        raise NotImplementedError("Domain subclass should reinitialize state.")


def build_environment(config: Dict[str, Any]) -> Environment:
    """Factory for the domain environment. TODO: implement once domain is chosen."""
    raise NotImplementedError
