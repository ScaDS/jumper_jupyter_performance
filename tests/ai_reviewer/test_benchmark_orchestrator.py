import logging
from unittest.mock import Mock

from jumper_extension.adapters.ai_reviewer.benchmark import fingerprint
from jumper_extension.adapters.ai_reviewer.benchmark.checks import CheckPlan, Decision
from jumper_extension.adapters.ai_reviewer.benchmark.models import FAILED, OK, RunOutcome
from jumper_extension.adapters.ai_reviewer.benchmark.replay import StrategyChanged
from jumper_extension.adapters.ai_reviewer.benchmark.orchestrator import (
    BASELINE_LABEL,
    BenchmarkOrchestrator,
)


def _syntax_only_checks() -> CheckPlan:
    """run off, validate_syntax on - the standalone syntax-check plan."""
    return CheckPlan(validate_syntax=Decision(True), run=Decision(False))

_PRINTS = {"y": {"kind": "scalar", "value": 42.0}}


def _runner(behaviour, log=None):
    """behaviour: code -> seconds, or an error string for a failed run."""
    runner = Mock()

    def run_once(code, tag, timeout=None):
        if log is not None:
            log.append(("run", tag))
        result = behaviour(code)
        if isinstance(result, str):
            return RunOutcome(status=FAILED, error=result)
        return RunOutcome(
            status=OK,
            duration_s=result,
            wall_s=result + 1.0,
            metrics={"cpu": {"mean": 50.0, "max": 90.0}},
            fingerprints=_PRINTS,
        )

    runner.run_once = run_once
    # These fakes stand in for the full replay, which is what a cross-check
    # would compare against, so there is nothing left to check.
    runner.cross_check_baseline = lambda code, tag="cross_check": None
    return runner


def _orchestrator(behaviour, fix_fn=None, log=None, **kwargs):
    return BenchmarkOrchestrator(
        runner=_runner(behaviour, log),
        fix_fn=fix_fn or (lambda code, error, label: code),
        runs=kwargs.pop("runs", 2),
        fix_attempts=kwargs.pop("fix_attempts", 3),
        **kwargs,
    )


def test_variant_is_scored_against_the_baseline():
    orchestrator = _orchestrator(lambda code: 4.0 if code == "base" else 1.0)

    results = orchestrator.run("base", [("1", "fast")])

    assert results[BASELINE_LABEL].duration_s == 4.0
    assert results["1"].speedup == 4.0
    assert results["1"].correctness == fingerprint.MATCH


def test_nothing_is_benchmarked_when_the_baseline_itself_will_not_run():
    orchestrator = _orchestrator(lambda code: "ImportError: no numpy")

    assert orchestrator.run("base", [("1", "fast")]) == {}


def test_a_failing_variant_is_repaired_and_then_measured():
    def behaviour(code):
        return 4.0 if code == "base" else (1.0 if code == "fixed" else "ValueError: boom")

    orchestrator = _orchestrator(behaviour, fix_fn=lambda code, error, label: "fixed")

    results = orchestrator.run("base", [("1", "broken")])

    assert results["1"].status == OK
    assert results["1"].attempts == 2
    assert orchestrator.final_code["1"] == "fixed"


def test_a_variant_is_given_up_on_after_its_fix_attempts():
    fix_calls = []

    def fix(code, error, label):
        fix_calls.append(error)
        return code

    orchestrator = _orchestrator(
        lambda code: 4.0 if code == "base" else "RuntimeError: nope",
        fix_fn=fix,
        fix_attempts=3,
    )

    results = orchestrator.run("base", [("1", "broken")])

    assert results["1"].status == FAILED
    assert "RuntimeError: nope" in results["1"].error
    assert len(fix_calls) == 3


def test_a_syntax_error_is_repaired_without_ever_being_run():
    log = []
    orchestrator = _orchestrator(
        lambda code: 4.0 if code == "base" else 1.0,
        fix_fn=lambda code, error, label: "y = 1",
        log=log,
    )

    orchestrator.run("base", [("1", "def broken(:")])

    variant_runs = [tag for kind, tag in log if kind == "run" and tag.startswith("1_")]
    assert variant_runs  # it ran after being fixed
    assert log[0][1].startswith(BASELINE_LABEL)  # but only the baseline ran before


def test_a_repair_that_returns_broken_syntax_costs_no_replay():
    executed = []

    def behaviour(code):
        executed.append(code)
        return 4.0 if code == "base" else 1.0

    repairs = iter(["def still_broken(:", "y = 1"])
    orchestrator = _orchestrator(
        behaviour,
        fix_fn=lambda code, error, label: next(repairs),
        fix_attempts=3,
    )

    orchestrator.run("base", [("1", "def broken(:")])

    # The variant went through three versions; the two that could not compile
    # were repaired without ever reaching a process.
    assert "def broken(:" not in executed
    assert "def still_broken(:" not in executed
    assert orchestrator.final_code["1"] == "y = 1"


# --- declared result names ---

def _binding_runner(behaviour, bindings: dict):
    """A runner where *bindings* maps a code string to the names it leaves.

    The ECCO shape: the payload binds the answers and a modulus, and a rewrite
    may keep either, both or neither.
    """
    runner = _runner(behaviour)
    original = runner.run_once

    def run_once(code, tag, timeout=None):
        outcome = original(code, tag, timeout)
        outcome.fingerprints = {
            name: {"kind": "scalar", "value": value}
            for name, value in bindings.get(code, {}).items()
        }
        return outcome

    runner.run_once = run_once
    runner.cross_check_baseline = lambda code, tag="cross_check": None
    return runner


_ECCO_BINDINGS = {
    "base": {"MOD": 998244353.0, "answers": 7.0},
    # Kept the answer, moved the modulus inside the function it now uses.
    "moved_constant": {"answers": 7.0},
    # Defined a faster solver and never called it.
    "never_called": {"MOD": 998244353.0},
}


def test_a_constant_that_moved_inside_a_function_no_longer_decides_the_verdict():
    orchestrator = BenchmarkOrchestrator(
        runner=_binding_runner(
            lambda code: 4.0 if code == "base" else 1.0, _ECCO_BINDINGS
        ),
        fix_fn=lambda code, error, label: code,
        runs=2,
        result_names=["answers"],
    )

    results = orchestrator.run("base", [("1", "moved_constant")])

    assert results["1"].correctness == fingerprint.MATCH
    assert results["1"].speedup == 4.0


def test_without_declared_names_a_moved_constant_still_costs_the_verdict():
    # The old behaviour, kept for every cell nobody described: nothing here can
    # tell a result from bookkeeping, so a dropped binding keeps the verdict
    # out of MATCH.
    orchestrator = BenchmarkOrchestrator(
        runner=_binding_runner(
            lambda code: 4.0 if code == "base" else 1.0, _ECCO_BINDINGS
        ),
        fix_fn=lambda code, error, label: code,
        runs=2,
    )

    results = orchestrator.run("base", [("1", "moved_constant")])

    assert results["1"].correctness == fingerprint.UNVERIFIED
    assert results["1"].missing_names == ["MOD"]


def test_a_variant_that_produced_no_result_is_failed_not_timed():
    fix_calls = []

    def fix(code, error, label):
        fix_calls.append(error)
        return code  # the model does not fix it

    orchestrator = BenchmarkOrchestrator(
        runner=_binding_runner(
            # Three microseconds, because it did not do the work.
            lambda code: 4.0 if code == "base" else 0.000003,
            _ECCO_BINDINGS,
        ),
        fix_fn=fix,
        runs=2,
        fix_attempts=1,
        result_names=["answers"],
    )

    results = orchestrator.run("base", [("1", "never_called")])

    # Not OK with a speedup of 1.3 million: nothing was measured.
    assert results["1"].status == FAILED
    assert results["1"].speedup is None
    assert "no longer produces the result" in results["1"].error
    assert "answers" in fix_calls[0]


def test_a_missing_result_is_repaired_and_then_measured():
    def fix(code, error, label):
        return "moved_constant"

    orchestrator = BenchmarkOrchestrator(
        runner=_binding_runner(
            lambda code: 4.0 if code == "base" else 1.0, _ECCO_BINDINGS
        ),
        fix_fn=fix,
        runs=2,
        result_names=["answers"],
    )

    results = orchestrator.run("base", [("1", "never_called")])

    assert results["1"].status == OK
    assert results["1"].correctness == fingerprint.MATCH
    assert results["1"].attempts == 2


def test_a_result_name_the_cell_never_binds_only_warns(caplog):
    orchestrator = BenchmarkOrchestrator(
        runner=_binding_runner(
            lambda code: 4.0 if code == "base" else 1.0, _ECCO_BINDINGS
        ),
        fix_fn=lambda code, error, label: code,
        runs=2,
        result_names=["nowhere"],
    )

    with caplog.at_level(logging.WARNING, logger="extension"):
        results = orchestrator.run("base", [("1", "moved_constant")])

    assert "binds no such name" in caplog.text
    # Nothing to compare, and nothing condemned for it either.
    assert results["1"].status == OK
    assert results["1"].correctness == fingerprint.UNVERIFIED


def _diverging_runner(behaviour, diverging_codes: set):
    """A runner whose *diverging_codes* come back with the wrong fingerprint."""
    runner = _runner(behaviour)
    original = runner.run_once

    def run_once(code, tag, timeout=None):
        outcome = original(code, tag, timeout)
        if code in diverging_codes:
            outcome.fingerprints = {"y": {"kind": "scalar", "value": 1.0}}
        return outcome

    runner.run_once = run_once
    # These fakes stand in for the full replay, which is what a cross-check
    # would compare against, so there is nothing left to check.
    runner.cross_check_baseline = lambda code, tag="cross_check": None
    return runner


def test_a_diverging_variant_is_repaired_rather_than_just_flagged():
    fix_calls = []

    def fix(code, error, label):
        fix_calls.append((label, error))
        return "correct"

    orchestrator = BenchmarkOrchestrator(
        runner=_diverging_runner(lambda code: 4.0 if code == "base" else 1.0, {"cheat"}),
        fix_fn=fix,
        runs=2,
    )

    results = orchestrator.run("base", [("1", "cheat")])

    assert results["1"].correctness == fingerprint.MATCH
    assert results["1"].attempts == 2
    assert orchestrator.final_code["1"] == "correct"
    assert fix_calls[0][0] == "option 1/1"
    assert "no longer computes the same result" in fix_calls[0][1]


def test_an_unrepairable_divergence_keeps_the_measurement_it_did_get():
    orchestrator = BenchmarkOrchestrator(
        runner=_diverging_runner(lambda code: 4.0 if code == "base" else 0.1, {"cheat"}),
        fix_fn=lambda code, error, label: "cheat",  # the model never fixes it
        runs=2,
        fix_attempts=2,
    )

    results = orchestrator.run("base", [("1", "cheat")])

    # Reporting nothing would be worse: the numbers are real, and the verdict
    # already says the speedup is unearned.
    assert results["1"].status == OK
    assert results["1"].speedup == 40.0
    assert results["1"].correctness == fingerprint.DIFFERS


def test_syntax_only_mode_validates_without_running_anything():
    log = []
    orchestrator = _orchestrator(
        lambda code: 1.0,
        log=log,
        checks=_syntax_only_checks(),
    )

    results = orchestrator.run("base", [("1", "y = 1")])

    assert log == []  # not even the baseline was replayed
    assert BASELINE_LABEL not in results  # nothing to compare against
    assert results["1"].status == OK
    assert results["1"].correctness == fingerprint.UNVERIFIED
    assert results["1"].duration_s is None
    assert orchestrator.final_code["1"] == "y = 1"


def test_syntax_only_mode_repairs_a_broken_suggestion_without_running_it():
    log = []
    orchestrator = _orchestrator(
        lambda code: 1.0,
        fix_fn=lambda code, error, label: "y = 1",
        log=log,
        checks=_syntax_only_checks(),
    )

    results = orchestrator.run("base", [("1", "def broken(:")])

    assert log == []  # the fix was checked statically, never executed
    assert results["1"].status == OK
    assert results["1"].attempts == 2
    assert orchestrator.final_code["1"] == "y = 1"


def test_syntax_only_mode_gives_up_on_an_unfixable_suggestion():
    orchestrator = _orchestrator(
        lambda code: 1.0,
        fix_fn=lambda code, error, label: "def still_broken(:",
        fix_attempts=2,
        checks=_syntax_only_checks(),
    )

    results = orchestrator.run("base", [("1", "def broken(:")])

    assert results["1"].status == FAILED
    assert "1" not in orchestrator.final_code


def test_nothing_runs_when_every_check_is_off():
    orchestrator = _orchestrator(
        lambda code: 1.0,
        checks=CheckPlan(validate_syntax=Decision(False), run=Decision(False)),
    )

    assert orchestrator.run("base", [("1", "y = 1")]) == {}


def test_timeout_budget_covers_the_prefix_and_room_to_be_slow():
    orchestrator = _orchestrator(lambda code: 4.0, timeout_factor=10.0)

    budget = orchestrator._timeout_from(
        [RunOutcome(status=OK, duration_s=4.0, wall_s=5.0)], 4.0
    )

    assert budget == 42.0  # 1s prefix x2, plus 4s x10


def test_a_strategy_giving_out_restarts_the_whole_benchmark(caplog):
    """Numbers from two replay modes must never end up in one report.

    A benchmark reports ratios, so a variant timed one way over a baseline timed
    the other is not a speedup. When the runner swaps strategies mid-run it says
    so, and everything measured before the swap has to be thrown away.
    """
    seen = []

    def run_once(code, tag, timeout=None):
        seen.append(tag)
        # Gives out once, partway through the first variant.
        if len([t for t in seen if t.startswith("1_")]) == 1 and "restarted" not in seen:
            seen.append("restarted")
            raise StrategyChanged("the zygote is gone")
        return RunOutcome(
            status=OK,
            duration_s=4.0 if code == "base" else 1.0,
            wall_s=5.0,
            metrics={},
            fingerprints=_PRINTS,
        )

    runner = Mock()
    runner.run_once = run_once
    runner.cross_check_baseline = lambda code, tag="cross_check": None
    orchestrator = BenchmarkOrchestrator(
        runner=runner,
        fix_fn=lambda code, error, label: code,
        runs=2,
        fix_attempts=3,
    )

    with caplog.at_level(logging.WARNING, logger="extension"):
        results = orchestrator.run("base", [("1", "fast")])

    assert results["1"].speedup == 4.0, "the surviving report is measured end to end"
    # The baseline is re-measured too: its own numbers came from the old mode.
    assert seen.count("baseline_0") == 2
    assert any("re-run on the full replay" in r.message for r in caplog.records)


def test_a_strategy_giving_out_twice_reports_nothing(caplog):
    """The full replay prepares nothing and cannot give out; twice means a defect."""
    runner = Mock()
    runner.run_once = Mock(side_effect=StrategyChanged("gone again"))
    orchestrator = BenchmarkOrchestrator(
        runner=runner,
        fix_fn=lambda code, error, label: code,
        runs=2,
        fix_attempts=3,
    )

    with caplog.at_level(logging.ERROR, logger="extension"):
        results = orchestrator.run("base", [("1", "fast")])

    assert results == {}
    assert any("gave out twice" in r.message for r in caplog.records)
