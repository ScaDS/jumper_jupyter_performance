"""Decide what each suggestion is worth, by running it.

Measuring is strictly one variant at a time - anything else would have them
compete for the very resources being measured, and the baseline was recorded
alone. Only the repair calls overlap: those are network waits, and letting one
variant's fix travel while another is being timed costs the measurement nothing.
"""
import logging
from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
from functools import partial
from typing import Callable

from jumper_extension.adapters.ai_reviewer.benchmark import fingerprint
from jumper_extension.adapters.ai_reviewer.benchmark.models import (
    FAILED,
    OK,
    BenchmarkResult,
    RunOutcome,
)
from jumper_extension.adapters.ai_reviewer.benchmark.checks import CheckPlan, all_active
from jumper_extension.adapters.ai_reviewer.benchmark.progress import BenchmarkProgress
from jumper_extension.adapters.ai_reviewer.benchmark.replay import StrategyChanged
from jumper_extension.adapters.ai_reviewer.benchmark.runner import BenchmarkRunner, median_of
from jumper_extension.adapters.ai_reviewer.language import LanguageAdapter, get_adapter

logger = logging.getLogger("extension")

BASELINE_LABEL = "baseline"

# How far the fast mode's baseline may sit from the full replay's before it is
# worth saying so. Wide on purpose: it is meant to catch state that was lost,
# not the ordinary cost of a mode or a busy machine.
_CROSS_CHECK_BAND = 2.0

# Below this, a ratio says nothing: a cell measured in tens of microseconds is
# dominated by first-call dispatch in whichever process happened to be colder,
# and both modes would trip the band on every fast cell. Results are still
# compared - that part is not noise-sensitive.
_CROSS_CHECK_FLOOR_S = 0.005


class _Candidate:
    def __init__(self, label: str, code: str, attempts_left: int):
        self.label = label
        self.code = code
        self.attempts_left = attempts_left
        self.attempts = 1
        self.error = ""
        # The last version that actually ran, kept so exhausted repairs fall
        # back on a real measurement instead of throwing it away.
        self.last_verdict: BenchmarkResult | None = None
        self.last_code = ""


class BenchmarkOrchestrator:
    """Times the baseline and every suggestion, repairing what fails."""

    def __init__(
        self,
        runner: BenchmarkRunner,
        fix_fn: Callable[[str, str, str], str],
        runs: int = 3,
        fix_attempts: int = 3,
        timeout_factor: float = 10.0,
        adapter: LanguageAdapter | None = None,
        checks: CheckPlan | None = None,
        cross_check: bool = True,
        result_names: tuple[str, ...] | list[str] | None = None,
    ):
        self.runner = runner
        self.fix_fn = fix_fn
        self.cross_check = cross_check
        # Which bindings carry the cell's result. Declared, because nothing
        # here can tell a cell's output from its bookkeeping: with the names
        # known, the verdict rests on them and a constant that moved inside a
        # function stops mattering; without them every binding counts, which
        # is the old behaviour and the only safe default.
        self.result_names = tuple(result_names or ())
        # Defaults to Python so direct constructors (and tests) keep working;
        # build_orchestrator passes the target cell's actual adapter.
        self.adapter = adapter or get_adapter("python")
        # Defaults to every step on, so nothing is skipped unless asked.
        self.checks = checks or all_active()
        self.runs = max(1, runs)
        self.fix_attempts = max(1, fix_attempts)
        self.timeout_factor = timeout_factor
        # label -> the code that was actually measured, once repairs settled
        self.final_code: dict[str, str] = {}
        self.progress = BenchmarkProgress(0, self.runs)
        self._position: dict[str, int] = {}

    def run(self, baseline_code: str, variants: list[tuple[str, str]]) -> dict:
        """Score each ``(label, code)`` variant the best the active checks allow.

        With the timed run on, this measures *baseline_code* and every variant
        and compares their results, returning ``{label: BenchmarkResult}`` with
        the baseline under ``BASELINE_LABEL``. With the run turned off but
        ``validate_syntax`` still on, it falls back to a syntax-only pass that
        still repairs broken suggestions - just without timing or comparison,
        and without a baseline entry. With neither active there is nothing to
        do. An empty dict also means the baseline itself would not run, leaving
        nothing to compare against.
        """
        try:
            if self.checks.run.active:
                return self._timed_on_one_instrument(baseline_code, variants)
            if self.checks.validate_syntax.active:
                return self._run_syntax_only(variants)
            # Both the run and the syntax gate are off: resolve_checks already
            # explained why, and there is genuinely nothing left to do here.
            logger.info(
                "[JUmPER]: benchmark has no active checks; nothing to measure or verify."
            )
            return {}
        finally:
            # A strategy may be holding a live process or a checkpoint on disk;
            # it outlives no single measurement, so this is where it is released.
            self.runner.close()

    def _timed_on_one_instrument(
        self,
        baseline_code: str,
        variants: list[tuple[str, str]],
    ) -> dict:
        """Measure everything, and if the strategy changes underneath, start over.

        A benchmark reports ratios, so every number in one has to come from the
        same instrument. When a strategy gives out mid-run the runner has already
        swapped in the full replay, but the measurements taken before the swap
        cannot be set beside the ones taken after: a variant timed one way over a
        baseline timed the other is not a speedup, it is the difference between
        two clocks.

        Worse, the budget each variant is given comes from how long the baseline
        took *including* its prefix. Under a mode that replays the prefix once,
        that share is near zero - so variants inheriting it after a swap would be
        killed by a timeout with no room for the prefix they now have to replay,
        and each one handed to the repair loop to fix code that was never wrong.

        Restarting is therefore the cheap option, not the expensive one. Once
        only: the full replay prepares nothing and cannot give out in turn, so a
        second swap would mean a defect here rather than a problem on the machine.
        """
        for attempt in (1, 2):
            try:
                return self._run_timed(baseline_code, variants)
            except StrategyChanged as change:
                if attempt == 2:
                    logger.error(
                        "[JUmPER]: the benchmark's replay strategy gave out twice "
                        f"({change}); giving up rather than reporting numbers from "
                        "two different instruments."
                    )
                    return {}
                logger.warning(
                    f"[JUmPER]: the benchmark's replay strategy gave out ({change}). "
                    "Everything measured so far is discarded and the whole benchmark "
                    "is being re-run on the full replay - measurements from two "
                    "different modes cannot be compared with each other, and this "
                    "way every number in the report comes from the same one. "
                    "Expect it to take longer."
                )
                self.final_code.clear()
        return {}

    def _run_timed(self, baseline_code: str, variants: list[tuple[str, str]]) -> dict:
        """The full benchmark: measure the baseline, then time and verify each variant."""
        self.progress = BenchmarkProgress(len(variants), self.runs)
        self._position = {label: index for index, (label, _) in enumerate(variants, start=1)}

        # Before the active strategy prepares anything, so the two never hold the
        # same prefix at the same time.
        reference = self._cross_check_reference(baseline_code)

        baseline_runs = self._measure(
            baseline_code,
            BASELINE_LABEL,
            timeout=None,
            on_run=self.progress.baseline_run,
        )
        if isinstance(baseline_runs, RunOutcome):
            logger.warning(_baseline_failure(baseline_runs))
            return {}

        duration, metrics = median_of(baseline_runs)
        self.progress.baseline_done(duration, _walls(baseline_runs))
        baseline = BenchmarkResult(
            label=BASELINE_LABEL,
            status=OK,
            duration_s=duration,
            metrics=metrics,
            correctness=fingerprint.MATCH,
        )
        prints = baseline_runs[-1].fingerprints
        self._check_declared_results(prints)
        self._compare_with_reference(reference, duration, prints)
        timeout = self._timeout_from(baseline_runs, duration)

        results = {BASELINE_LABEL: baseline}
        results.update(
            self._drive(
                variants,
                partial(
                    self._process_timed,
                    timeout=timeout,
                    baseline_duration=duration,
                    baseline_prints=prints,
                ),
            )
        )
        self.progress.finished()
        return results

    def _cross_check_reference(self, baseline_code: str):
        """One baseline measured through the full replay, to check the fast mode by.

        Returns None when the check is off or the active strategy is already the
        full replay.
        """
        if not self.cross_check:
            return None
        return self.runner.cross_check_baseline(baseline_code)

    def _compare_with_reference(self, reference, duration: float, prints: dict):
        """Hold the fast mode's baseline against the full replay's.

        Results that differ mean the mode rebuilt state the cell notices, and
        nothing inside the benchmark would ever say so - every variant is
        compared against this same wrongly rebuilt baseline. That is worth
        discarding the run for: the strategy is swapped and the benchmark starts
        again on the full replay.

        Durations are treated far more gently. A restored process misses the
        warm-up the prefix would have done, a forked one pays for memory it
        inherited, and machines are noisy; a band this wide catches state that
        was lost, not the ordinary cost of a mode.
        """
        if reference is None:
            return

        if not reference.ok:
            # The check was asked for and could not be made. Carrying on would
            # report numbers whose state nothing ever verified, so the mode goes
            # back to the one that needs no verifying.
            reason = f"its baseline could not be measured through the full replay: {reference.error}"
            self.runner.fall_back(reason)
            raise StrategyChanged(reason)

        verdict, differing, missing = fingerprint.compare_all(
            reference.fingerprints, prints
        )
        if differing or missing:
            # The same code ran both ways. Values that differ mean the mode
            # rebuilt the wrong state; names that are absent under one mode and
            # not the other mean the same thing, and are no more acceptable.
            names = ", ".join(differing + missing) or "unnamed values"
            reason = (
                f"its baseline computed something else than the full replay did ({names})"
            )
            self.runner.fall_back(reason)
            raise StrategyChanged(reason)
        if verdict == fingerprint.UNVERIFIED:
            # Common and not an error: a cell that mutates in place or only has
            # side effects binds nothing comparable. Worth saying plainly, since
            # the guide sells this check as what keeps a fast mode honest.
            logger.warning(
                "[JUmPER]: the cell under review binds nothing this benchmark can "
                "compare, so the replay mode's rebuilt state could not be checked "
                "against a full replay. The timings below are unverified in that "
                "sense; --replay-mode full needs no such check."
            )

        reference_duration = reference.duration_s or 0.0
        if min(reference_duration, duration or 0.0) < _CROSS_CHECK_FLOOR_S:
            return
        ratio = duration / reference_duration
        if 1 / _CROSS_CHECK_BAND <= ratio <= _CROSS_CHECK_BAND:
            return
        logger.warning(
            f"[JUmPER]: the baseline takes {duration}s under this replay mode against "
            f"{reference_duration}s under the full replay ({ratio:.1f}x). Their results "
            "matched - as far as statistical signatures can tell - but the timings are "
            "not comparable with a full replay's: a restored process has none of the "
            "warm-up the prefix would have done, and a forked one pays for the memory "
            "it inherited."
        )

    def _run_syntax_only(self, variants: list[tuple[str, str]]) -> dict:
        """Check each suggestion parses, repairing what does not - no timing.

        The timed run is off, so there is nothing to measure or fingerprint, but
        a broken suggestion can still be caught and handed to the same repair
        loop. Each surviving variant is reported OK with ``UNVERIFIED``
        correctness and its valid (possibly repaired) code in ``final_code``;
        ones that never parse are reported FAILED.
        """
        logger.info(
            f"[JUmPER]: benchmark: validating the syntax of {len(variants)} "
            "suggestion(s), with the timed run off."
        )
        self.progress = BenchmarkProgress(len(variants), self.runs)
        self._position = {label: index for index, (label, _) in enumerate(variants, start=1)}

        results = self._drive(variants, self._process_syntax)

        valid = sum(1 for result in results.values() if result.ok)
        logger.info(
            f"[JUmPER]: benchmark: syntax validated - {valid} valid, "
            f"{len(results) - valid} unfixable."
        )
        return results

    def _drive(self, variants: list[tuple[str, str]], process: Callable) -> dict:
        """Run *process* over each variant, draining repairs the shared way.

        Every mode differs only in what it does with a fresh candidate; the
        repair round-trip - fold a returned fix back into the queue, or settle a
        candidate whose repair produced nothing - is identical, so it lives here
        once. *process* is called as ``process(results, candidate, pending,
        repairing, pool)`` and is responsible for settling the candidate or
        submitting it for repair.
        """
        results: dict = {}
        pending = [_Candidate(label, code, self.fix_attempts) for label, code in variants]
        repairing: dict[Future, _Candidate] = {}

        pool = ThreadPoolExecutor(max_workers=max(1, len(pending)))
        try:
            while pending or repairing:
                if pending:
                    process(results, pending.pop(0), pending, repairing, pool)
                    continue

                done, _ = wait(list(repairing), return_when=FIRST_COMPLETED)
                for future in done:
                    candidate = repairing.pop(future)
                    fixed = _fixed_code(future, candidate)
                    if fixed is None:
                        self._settle(results, candidate)
                        continue
                    candidate.code = fixed
                    candidate.attempts += 1
                    pending.append(candidate)
        except BaseException:
            # Do not wait out a repair on the way out. Nothing here will use its
            # answer, and an unfinished model call can hold the door for minutes.
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)
        return results

    def _process_timed(
        self,
        results: dict,
        candidate: _Candidate,
        pending: list,
        repairing: dict,
        pool: ThreadPoolExecutor,
        *,
        timeout: float,
        baseline_duration: float,
        baseline_prints: dict,
    ) -> None:
        """Time one candidate against the baseline, repairing a crash or a divergence."""
        position = self._position[candidate.label]

        # Syntax is checkable for free, so broken code never costs a replay -
        # whether it came from the model or from a repair.
        if not self._syntax_ok(candidate):
            self._reject(results, candidate, position, pool, repairing)
            return

        outcome = self._measure(
            candidate.code,
            candidate.label,
            timeout,
            on_run=lambda index: self.progress.variant_run(position, index),
        )
        outstanding = len(pending) + len(repairing)
        if isinstance(outcome, list):
            lost = self._lost_results(baseline_prints, outcome[-1].fingerprints)
            if lost:
                # It ran, and it produced none of what the cell is for. This is
                # not a slow or a wrong measurement, it is no measurement: the
                # timed region did not do the work, so its duration is not a
                # speedup of anything and must not be settled as one.
                candidate.error = fingerprint.describe_missing_results(lost)
                self._reject(results, candidate, position, pool, repairing)
                return
            verdict = self._verdict(candidate, outcome, baseline_duration, baseline_prints)
            candidate.last_verdict = verdict
            candidate.last_code = candidate.code

            # A wrong answer is worth repairing too: it costs the review more
            # than a crash, since nothing else catches it.
            if verdict.correctness == fingerprint.DIFFERS:
                candidate.error = fingerprint.describe_divergence(
                    baseline_prints,
                    outcome[-1].fingerprints,
                    verdict.differing_names,
                )
                if self._submit_fix(candidate, pool, repairing):
                    self.progress.variant_diverged(
                        position,
                        verdict.differing_names,
                        candidate.attempts,
                        self.fix_attempts,
                    )
                    return

            self._settle(results, candidate, verdict)
            self.progress.variant_done(
                position,
                _summarize(verdict),
                _walls(outcome),
                outstanding=outstanding,
            )
        else:
            candidate.error = outcome.error
            self._reject(results, candidate, position, pool, repairing)

    def _process_syntax(
        self,
        results: dict,
        candidate: _Candidate,
        pending: list,
        repairing: dict,
        pool: ThreadPoolExecutor,
    ) -> None:
        """Parse one candidate, repairing it if it does not - never running it."""
        position = self._position[candidate.label]
        if self._syntax_ok(candidate):
            self.final_code[candidate.label] = candidate.code
            results[candidate.label] = BenchmarkResult(
                label=candidate.label,
                status=OK,
                attempts=candidate.attempts,
                correctness=fingerprint.UNVERIFIED,
            )
            self.progress.variant_validated(position, candidate.attempts)
        else:
            self._reject(results, candidate, position, pool, repairing)

    def _reject(
        self,
        results: dict,
        candidate: _Candidate,
        position: int,
        pool: ThreadPoolExecutor,
        repairing: dict,
    ) -> None:
        """Hand a candidate that did not work back for repair, or give up on it."""
        if self._submit_fix(candidate, pool, repairing):
            self.progress.variant_failed(
                position,
                candidate.error,
                candidate.attempts,
                self.fix_attempts,
            )
        else:
            self._settle(results, candidate)
            self.progress.variant_gave_up(position, self.fix_attempts)

    def _settle(
        self,
        results: dict,
        candidate: _Candidate,
        verdict: BenchmarkResult | None = None,
    ) -> None:
        """Record what a candidate is finally worth.

        Once repairs are spent, a version that ran - even one whose results
        drifted - beats reporting nothing: the numbers are real, and the card
        says plainly that the speedup was not earned.
        """
        final = verdict or candidate.last_verdict
        if final is None:
            results[candidate.label] = _failed(candidate)
            return
        self.final_code[candidate.label] = candidate.last_code or candidate.code
        results[candidate.label] = final

    def _submit_fix(self, candidate: _Candidate, pool: ThreadPoolExecutor, repairing: dict) -> bool:
        """Queue a repair for *candidate*; False once its attempts are spent."""
        if candidate.attempts_left <= 0:
            return False
        candidate.attempts_left -= 1
        future = pool.submit(
            self.fix_fn,
            candidate.code,
            candidate.error,
            self._label_of(candidate),
        )
        repairing[future] = candidate
        return True

    def _label_of(self, candidate: _Candidate) -> str:
        """How this candidate is named to the user, so logs line up with the card."""
        position = self._position.get(candidate.label)
        if position is None:
            return f"option {candidate.label}"
        return f"option {position}/{self.progress.total_variants}"

    def _compared_scope(self, baseline_prints: dict, prints: dict) -> tuple[dict, dict]:
        """The bindings this verdict rests on.

        With result names declared, only those. Everything else a cell happens
        to bind is bookkeeping - a modulus, a bound, a counter - and a rewrite
        is free to move it inside a function; that is what deleting the class
        which read it looks like. Letting such a name decide turns a correct
        rewrite into an unverified one, which is a worse error than ignoring it:
        the result itself is still compared, and it is the only thing a user
        cares about. Undeclared, nothing can tell output from bookkeeping, so
        every binding counts.
        """
        if not self.result_names:
            return baseline_prints, prints
        declared = set(self.result_names)
        return (
            {
                name: value
                for name, value in baseline_prints.items()
                if name in declared
            },
            {name: value for name, value in prints.items() if name in declared},
        )

    def _lost_results(self, baseline_prints: dict, prints: dict) -> list[str]:
        """Declared result names the baseline bound and this variant did not.

        Limited to what the baseline actually produced, so a name declared for
        a cell that never binds it cannot condemn every rewrite of it.
        """
        return [
            name
            for name in self.result_names
            if name in baseline_prints and name not in prints
        ]

    def _check_declared_results(self, baseline_prints: dict) -> None:
        """Say so when a declared result name is not in the baseline at all.

        Silence here would be the worst outcome: the verdict would quietly rest
        on nothing, and every variant would read as unverified for a reason
        nobody could see.
        """
        absent = [
            name for name in self.result_names if name not in baseline_prints
        ]
        if absent:
            logger.warning(
                "[JUmPER]: the benchmark was told the cell's result is "
                f"{', '.join(absent)}, but the cell under review binds no such "
                "name at its top level. Correctness will rest on whatever else "
                "it binds, as if no result name had been named."
            )

    def _syntax_ok(self, candidate: _Candidate) -> bool:
        if not self.checks.validate_syntax.active:
            return True
        result = self.adapter.validate_syntax(candidate.code)
        if result.ok:
            return True
        candidate.error = result.error
        return False

    def _measure(self, code: str, label: str, timeout: float | None, on_run=None):
        """Time *code* ``runs`` times, or return the outcome that stopped it."""
        outcomes = []
        for index in range(self.runs):
            if on_run is not None:
                on_run(index)
            outcome = self.runner.run_once(code, tag=f"{label}_{index}", timeout=timeout)
            if not outcome.ok:
                return outcome
            outcomes.append(outcome)
        return outcomes

    def _timeout_from(self, baseline_runs: list[RunOutcome], duration: float) -> float:
        """Budget a variant gets: the prefix it must replay, plus room to be slow."""
        walls = [run.wall_s for run in baseline_runs if run.wall_s]
        prefix = max(walls) - duration if walls else 0.0
        return max(prefix, 0.0) * 2 + max(duration * self.timeout_factor, 5.0)

    def _verdict(
        self,
        candidate: _Candidate,
        outcomes: list[RunOutcome],
        baseline_duration: float,
        baseline_prints: dict,
    ) -> BenchmarkResult:
        duration, metrics = median_of(outcomes)
        baseline_scope, variant_scope = self._compared_scope(
            baseline_prints, outcomes[-1].fingerprints
        )
        correctness, differing, missing = fingerprint.compare_all(
            baseline_scope, variant_scope
        )
        return BenchmarkResult(
            label=candidate.label,
            status=OK,
            attempts=candidate.attempts,
            duration_s=duration,
            metrics=metrics,
            speedup=round(baseline_duration / duration, 2) if duration else None,
            correctness=correctness,
            differing_names=differing,
            missing_names=missing,
        )


def _baseline_failure(outcome: RunOutcome) -> str:
    """Say what could not be replayed, and whose problem it is.

    A benchmark measures the cell under review against its own predecessors, so
    it needs both. When the prefix is what failed, the cell under review has not
    been run at all - saying it "did not replay cleanly" points a user, and a
    repair loop, at code that is very likely fine.
    """
    if outcome.prefix_cell is None:
        return (
            "[JUmPER]: the cell under review did not replay cleanly, so there is "
            f"nothing to compare against: {outcome.error}"
        )
    return (
        f"[JUmPER]: benchmark skipped: prefix cell {outcome.prefix_cell} could not be "
        f"replayed - {_last_line(outcome.error)}\n"
        "A replay runs in a plain interpreter, so a cell that talks to the notebook "
        "frontend, or to anything else only a live session provides, cannot be "
        "reproduced. The review itself is unaffected.\n"
        f"{outcome.error}"
    )


def _last_line(error: str) -> str:
    """The exception line of a traceback - the part worth putting in a sentence."""
    lines = [line.strip() for line in (error or "").splitlines() if line.strip()]
    return lines[-1] if lines else "no error was reported"


def _walls(outcomes: list[RunOutcome]) -> list[float]:
    return [outcome.wall_s for outcome in outcomes if outcome.wall_s]


def _summarize(verdict: BenchmarkResult) -> str:
    """The one-line gist of a measured variant, for the progress log."""
    parts = [f"{verdict.duration_s}s"]
    if verdict.speedup:
        parts.append(
            f"{verdict.speedup}x faster" if verdict.speedup >= 1
            else f"{round(1 / verdict.speedup, 2)}x slower"
        )
    if verdict.correctness == fingerprint.DIFFERS:
        parts.append("but results differ")
    elif verdict.missing_names:
        parts.append(f"no longer binds {', '.join(verdict.missing_names)}")
    elif verdict.correctness == fingerprint.UNVERIFIED:
        parts.append("results unverified")
    if verdict.attempts > 1:
        parts.append(f"repaired {verdict.attempts - 1}x")
    return ", ".join(parts)


def _fixed_code(future: Future, candidate: _Candidate) -> str | None:
    try:
        fixed = future.result()
    except Exception as error:
        logger.debug("[JUmPER]: repair call failed for %s: %s", candidate.label, error)
        return None
    return fixed or None


def _failed(candidate: _Candidate) -> BenchmarkResult:
    return BenchmarkResult(
        label=candidate.label,
        status=FAILED,
        attempts=candidate.attempts,
        error=candidate.error,
    )
