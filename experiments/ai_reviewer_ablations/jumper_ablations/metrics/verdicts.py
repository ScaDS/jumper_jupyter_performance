"""Reading the benchmark's verdicts the same way everywhere.

The reviewer reports a status, a fingerprint comparison and a speedup, and
these three have to be combined carefully. A suggestion that ran and got
faster while computing something else is not a win, and a suggestion whose
results could not be compared is not a win either - it is an unknown. Every
metric that says "correct" means the definition in :func:`is_correct`, and it
means it in one place so that no metric can quietly relax it.
"""
from __future__ import annotations

from jumper_ablations.records.schema import BenchmarkRecord, RunRecord

# The reviewer's fingerprint verdicts. `match` is what the experiment plan
# calls `verified`.
VERIFIED = "match"
DIFFERS = "differs"
UNVERIFIED = "unverified"

STATUS_OK = "ok"


def verdicts_by_index(record: RunRecord) -> dict[int, BenchmarkRecord]:
    """Benchmark rows keyed by 1-based suggestion index."""
    return record.outputs.verdicts()


def ordered_verdicts(record: RunRecord) -> list[BenchmarkRecord]:
    """Verdicts in the order the model ranked the suggestions."""
    by_index = verdicts_by_index(record)
    return [by_index[index] for index in sorted(by_index)]


def ran(verdict: BenchmarkRecord) -> bool:
    """The variant executed and was timed."""
    return verdict.status == STATUS_OK and verdict.duration_s is not None


def is_correct(verdict: BenchmarkRecord) -> bool:
    """It ran, it was compared, and it produced the same results.

    `unverified` is deliberately not correct. Counting it as correct would
    reward exactly the rewrites whose results the harness could not check.
    """
    return ran(verdict) and verdict.correctness == VERIFIED


def speedup_of(verdict: BenchmarkRecord) -> float | None:
    if not ran(verdict):
        return None
    return verdict.speedup


def correct_speedups(record: RunRecord) -> list[float]:
    return [
        verdict.speedup
        for verdict in ordered_verdicts(record)
        if is_correct(verdict) and verdict.speedup is not None
    ]


def measured_speedups(record: RunRecord) -> list[float]:
    return [
        verdict.speedup
        for verdict in ordered_verdicts(record)
        if ran(verdict) and verdict.speedup is not None
    ]


def suggestion_codes(record: RunRecord) -> dict[int, str]:
    return {
        suggestion.index: suggestion.code
        for suggestion in record.outputs.suggestions
    }


def normalised_code(code: str) -> str:
    """Code with formatting differences removed.

    Used to tell a repair that rewrote the approach from one that only moved
    whitespace around. It is a lower bound on "the idea changed" - a rename is
    still counted as a change - and the metric says so.
    """
    lines = [line.strip() for line in (code or "").splitlines()]
    return "\n".join(line for line in lines if line)
