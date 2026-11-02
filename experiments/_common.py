"""Shared helpers for experiment scripts."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents import AGENT_REGISTRY, build_backend  # noqa: E402
from src.environment import build_environment  # noqa: E402
from src.workflow import Workflow  # noqa: E402


def load_config(path: str = "configs/experiment.yaml") -> Dict[str, Any]:
    with open(ROOT / path) as f:
        cfg = yaml.safe_load(f)
    random.seed(cfg.get("seed", 0))
    return cfg


def load_tasks(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    with open(ROOT / cfg["workflow"]["tasks_file"]) as f:
        return json.load(f)["tasks"]


def role_llm_cfg(cfg: Dict[str, Any], role: str) -> Dict[str, Any]:
    """Merge llm.defaults with llm.roles.<role>."""
    llm = cfg["llm"]
    return {**llm.get("defaults", {}), **(llm.get("roles", {}).get(role) or {})}


def build_agent(cfg: Dict[str, Any], role: str):
    return AGENT_REGISTRY[role](build_backend(role_llm_cfg(cfg, role)))


def build_workflow(cfg: Dict[str, Any]) -> Workflow:
    env = build_environment(cfg)
    agents = [build_agent(cfg, s) for s in cfg["workflow"]["stages"]]
    return Workflow(agents, env)


def build_strategy(cfg: Dict[str, Any], name: str):
    """Recovery methods by config name (all get the oracle fault identity)."""
    from src.agents import Verifier
    from src.recovery import FullRestart, VerifyAll
    if name == "full_restart":
        return FullRestart(cfg.get("environment", {}).get("restore_mode", "realistic"))
    if name == "full_restart_oracle":
        return FullRestart("oracle")
    if name == "verify_all":
        backend = build_backend(role_llm_cfg(cfg, "verifier"))
        return VerifyAll(Verifier(backend, tool_access=cfg["recovery"].get("verifier_tool_access", True)))
    raise KeyError(f"unknown recovery method {name}")


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
