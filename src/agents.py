"""Agents and LLM backends.

Each agent consumes upstream artifacts (and optionally shared state), calls
an LLM backend, and emits a new Artifact whose `sources` list its inputs.
The backend is pluggable so the same workflow runs on a deterministic mock
(for reproducible experiments) or a real model.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .environment import Environment
from .tracing import Artifact, LLMCall, Tracer, new_id


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------
class LLMBackend(ABC):
    @abstractmethod
    def complete(self, system: str, prompt: str) -> Dict[str, Any]:
        """Return {"text": str, "prompt_tokens": int, "completion_tokens": int}."""


class MockLLM(LLMBackend):
    """Deterministic stand-in for development. TODO: scripted responses per stage."""

    def complete(self, system: str, prompt: str) -> Dict[str, Any]:
        raise NotImplementedError


class OpenAICompatLLM(LLMBackend):
    """Any OpenAI-compatible endpoint (UCSD course API, Ollama's /v1, vLLM, ...).

    Reads OPENAI_API_KEY / OPENAI_BASE_URL from the environment (or a local
    .env file, which is git-ignored). Never hard-code the key.
    """

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1024,
                 base_url: Optional[str] = None, api_key: Optional[str] = None,
                 timeout_s: float = 60.0):
        import os
        from openai import OpenAI
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        self.client = OpenAI(
            api_key=api_key or os.environ["OPENAI_API_KEY"],
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            timeout=timeout_s,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, system: str, prompt: str) -> Dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        usage = resp.usage
        return {
            "text": resp.choices[0].message.content or "",
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        }


def build_backend(cfg: Dict[str, Any]) -> LLMBackend:
    backend = cfg.get("backend", "mock")
    if backend == "mock":
        return MockLLM()
    if backend in ("openai", "ucsd"):
        return OpenAICompatLLM(model=cfg["model"], temperature=cfg.get("temperature", 0.0),
                               max_tokens=cfg.get("max_tokens", 1024))
    if backend == "ollama":  # Ollama exposes an OpenAI-compatible API
        return OpenAICompatLLM(model=cfg["model"], temperature=cfg.get("temperature", 0.0),
                               max_tokens=cfg.get("max_tokens", 1024),
                               base_url=cfg.get("base_url", "http://localhost:11434/v1"),
                               api_key="ollama")
    raise NotImplementedError(f"LLM backend '{backend}' not implemented yet")


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
class Agent(ABC):
    name: str = "agent"
    system_prompt: str = ""

    def __init__(self, llm: LLMBackend):
        self.llm = llm

    def call_llm(self, prompt: str, tracer: Tracer, purpose: str = "work") -> str:
        t0 = time.time()
        out = self.llm.complete(self.system_prompt, prompt)
        tracer.add_llm_call(LLMCall(
            call_id=new_id("llm"), stage=self.name,
            prompt_tokens=out.get("prompt_tokens", 0),
            completion_tokens=out.get("completion_tokens", 0),
            latency_s=time.time() - t0, purpose=purpose,
        ))
        return out["text"]

    @abstractmethod
    def run(self, task: Dict[str, Any], inputs: List[Artifact],
            env: Environment, tracer: Tracer) -> Artifact:
        """Produce this stage's artifact from upstream inputs."""


class Researcher(Agent):
    name = "researcher"
    system_prompt = "TODO"

    def run(self, task, inputs, env, tracer):
        raise NotImplementedError


class Planner(Agent):
    name = "planner"
    system_prompt = "TODO"

    def run(self, task, inputs, env, tracer):
        raise NotImplementedError


class Executor(Agent):
    name = "executor"
    system_prompt = "TODO"

    def run(self, task, inputs, env, tracer):
        raise NotImplementedError


class Verifier(Agent):
    """Used by the always-verify baseline to check an intermediate artifact."""
    name = "verifier"
    system_prompt = "TODO"

    def run(self, task, inputs, env, tracer):
        raise NotImplementedError


AGENT_REGISTRY = {cls.name: cls for cls in (Researcher, Planner, Executor, Verifier)}
