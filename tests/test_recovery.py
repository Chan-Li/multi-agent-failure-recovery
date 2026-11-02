"""Block 5: recovery baselines on the mock backend."""
import copy
import json
from pathlib import Path

import pytest

from src.agents import Executor, MockLLM, Planner, Researcher, Verifier
from src.environment import CourseEnrollmentEnv
from src.fault_injection import FaultInjector, FaultSpec
from src.recovery import FullRestart, VerifyAll
from src.workflow import Workflow

ROOT = Path(__file__).resolve().parents[1]
TASKS = {t["task_id"]: t for t in json.loads((ROOT / "data/tasks.json").read_text())["tasks"]}
M = MockLLM()


def wf(competitors=True):
    return Workflow([Researcher(M), Planner(M), Executor(M)], CourseEnrollmentEnv(competitors=competitors))


def spec_for(task, fid):
    kind = {"researcher-wrong-units": "incorrect_result", "researcher-wrong-seats": "incorrect_result",
            "researcher-injection": "malicious_instruction", "planner-injection": "malicious_instruction",
            "researcher-timeout": "timeout"}[fid]
    cat = "adversarial" if "injection" in fid else "accidental"
    stage = "planner" if fid.startswith("planner") else "researcher"
    return FaultSpec(fid, kind, cat, stage)


def faulty(task, fid, competitors=True):
    w = wf(competitors)
    spec = spec_for(task, fid)
    res = w.run(task, injector=FaultInjector(spec, task), run_id=f"f-{fid}")
    fa = next((a.artifact_id for a in res.tracer.artifacts.values() if a.faulty), None)
    return w, res, fa, spec


def fork(w):
    """Each method recovers from its own copy of the faulty world."""
    return Workflow(w.agents, copy.deepcopy(w.env))


# 5.1 start snapshot
def test_start_snapshot_recorded():
    res = wf().run(TASKS["t000"])
    assert res.metrics["start_snapshot"].startswith("start-")


# 5.2 full restart
def test_full_restart_benign_fault_recovers():
    t = TASKS["t017"]  # wrong-seats is benign here
    w, res, fa, spec = faulty(t, "researcher-wrong-seats")
    rep = FullRestart().recover(fork(w), t, res, fa, spec)
    assert rep.recovered_success and rep.clean_recovery
    assert rep.recomputed_stages == ["researcher", "planner", "executor"] and rep.extra_llm_tokens > 0


def test_full_restart_realistic_cannot_undo_unneeded_drops():
    t = TASKS["t017"]  # wrong units -> drops both electives; waitlisters take the seats
    w, res, fa, spec = faulty(t, "researcher-wrong-units")
    real = FullRestart("realistic").recover(fork(w), t, res, fa, spec)
    orac = FullRestart("oracle").recover(fork(w), t, res, fa, spec)
    assert len(real.harmful_irreversible) == 1           # the drop that was not needed
    assert real.persisted_events >= 2 and not real.clean_recovery
    assert not real.recovered_success                    # lost elective is gone for good
    assert orac.recovered_success and orac.clean_recovery and orac.persisted_events == 0


def test_full_restart_injection_leaves_leaked_email():
    t = TASKS["t000"]
    w, res, fa, spec = faulty(t, "researcher-injection")
    rep = FullRestart().recover(fork(w), t, res, fa, spec)
    leaked = [a for a in rep.harmful_irreversible if a.tool == "send_email"]
    assert leaked and not rep.clean_recovery


def test_full_restart_after_timeout():
    t = TASKS["t000"]
    w, res, fa, spec = faulty(t, "researcher-timeout")
    assert fa is None and not res.success
    rep = FullRestart().recover(fork(w), t, res, fa, spec)
    assert rep.recovered_success and rep.irreversible_actions == []


# 5.3 verify-all
@pytest.mark.parametrize("fid", ["researcher-wrong-units", "researcher-injection", "planner-injection"])
def test_verify_all_catches_fault_before_action(fid):
    t = TASKS["t017"]
    w, res, fa, spec = faulty(t, fid, competitors=False)
    rep = VerifyAll(Verifier(M)).recover(fork(w), t, res, fa, spec)
    assert rep.detected is True and rep.false_alarms == 0
    assert rep.recovered_success and rep.clean_recovery and rep.irreversible_actions == []
    assert rep.recomputed_stages == [spec.stage] and rep.extra_llm_tokens > 0


def test_verify_all_latency_lets_competitor_take_seat():
    """Same detection, but the regeneration costs a world tick: competitor C
    arrives at t=3 and takes the last seat of the planned SYS367-A."""
    t = TASKS["t017"]
    w, res, fa, spec = faulty(t, "researcher-wrong-units")
    rep = VerifyAll(Verifier(M)).recover(fork(w), t, res, fa, spec)
    assert rep.detected and not rep.recovered_success and rep.harmful_irreversible == []
    assert any("systems" in f for f in rep.failures)


def test_full_restart_discards_valid_work_in_moving_world():
    """Benign fault: the faulty run already succeeded. Restarting rolls back
    the valid enrollments, competitors arrive meanwhile, rerun can fail."""
    lost = 0
    for t in TASKS.values():
        if "researcher-wrong-seats" not in t["valid_faults"]:
            continue
        w, res, fa, spec = faulty(t, "researcher-wrong-seats")
        if res.success and not FullRestart().recover(fork(w), t, res, fa, spec).recovered_success:
            lost += 1
    assert lost >= 1


# 5.4 verify-all on a clean run: cost, no false alarms (mock verifier is exact)
def test_verify_all_clean_overhead():
    t = TASKS["t000"]
    w = wf()
    plain = w.run(t)
    res = w.run(t, verifier=Verifier(M))
    assert res.success and res.metrics["regenerated"] == []
    assert all(v["ok"] for v in res.metrics["verdicts"])
    verify_tok = sum(c.prompt_tokens + c.completion_tokens for c in res.tracer.llm_calls if c.purpose == "verify")
    assert verify_tok > 0
    assert len(res.tracer.actions) > len(plain.tracer.actions)   # verifier's registrar reads
