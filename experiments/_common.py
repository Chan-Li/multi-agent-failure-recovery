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


def build_workflow(cfg: Dict[str, Any]) -> Workflow:
    llm = build_backend(cfg["llm"])
    env = build_environment(cfg)
    agents = [AGENT_REGISTRY[s](llm) for s in cfg["workflow"]["stages"]]
    return Workflow(agents, env)


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
