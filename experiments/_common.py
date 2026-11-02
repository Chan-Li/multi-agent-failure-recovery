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


def build_verifier(cfg: Dict[str, Any], self_verify: bool = False):
    """Strong verifier (llm.roles.verifier) or self-verification (the planner's model)."""
    from src.agents import Verifier
    llm_cfg = role_llm_cfg(cfg, "verifier")
    if self_verify:
        llm_cfg = {**llm_cfg, "model": role_llm_cfg(cfg, "planner")["model"]}
    return Verifier(build_backend(llm_cfg), tool_access=cfg["recovery"].get("verifier_tool_access", True))


def build_strategy(cfg: Dict[str, Any], name: str):
    """Recovery methods by config name (all get the oracle fault identity)."""
    from src.agents import Verifier
    from src.recovery import FullRestart, VerifyAll
    if name == "full_restart":
        return FullRestart(cfg.get("environment", {}).get("restore_mode", "realistic"))
    if name == "full_restart_oracle":
        return FullRestart("oracle")
    if name in ("verify_all", "verify_all_self"):
        return VerifyAll(build_verifier(cfg, self_verify=name == "verify_all_self"), label=name)
    raise KeyError(f"unknown recovery method {name}")


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def run_jobs(jobs: List[Dict[str, Any]], fn, out_path: Path, workers: int = 1,
             fresh: bool = False) -> int:
    """Run fn(job) -> row for every job, `workers` at a time, appending each
    row to `out_path` as soon as it finishes. Resumable: jobs whose run_id is
    already in the file are skipped (pass fresh=True to start over).
    Returns the number of new rows."""
    import threading
    import time
    import traceback
    from concurrent.futures import ThreadPoolExecutor, as_completed

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fresh and out_path.exists():
        out_path.unlink()
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["run_id"])
    todo = [j for j in jobs if j["run_id"] not in done]
    print(f"{len(jobs)} jobs, {len(done)} already done, {len(todo)} to run, {workers} workers", flush=True)
    lock, n, t0 = threading.Lock(), 0, time.time()

    def safe(job):
        try:
            return fn(job)
        except Exception as e:  # never lose the whole batch to one bad job
            return {"run_id": job["run_id"], "job_error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc()[-2000:]}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(safe, j) for j in todo]
        for fut in as_completed(futures):
            row = fut.result()
            with lock:
                with open(out_path, "a") as f:
                    f.write(json.dumps(row, default=str) + "\n")
                n += 1
                if n % 10 == 0 or n == len(todo):
                    el = time.time() - t0
                    print(f"  {n}/{len(todo)} done, {el / 60:.1f} min elapsed, "
                          f"~{el / n * (len(todo) - n) / 60:.1f} min left", flush=True)
    return n
