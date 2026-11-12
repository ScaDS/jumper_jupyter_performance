from __future__ import annotations

import dataclasses

OK = "ok"
FAILED = "failed"
TIMEOUT = "timeout"
# What an attempt can come to besides the replay statuses above: code that does
# not parse, and code that ran without binding the result the cell is for.
INVALID_SYNTAX = "invalid_syntax"
MISSING_RESULTS = "missing_results"


@dataclasses.dataclass
class RunOutcome:
    """One replay of a cell: either it produced numbers, or it did not."""
    status: str
    duration_s: float | None = None
    wall_s: float | None = None
    metrics: dict = dataclasses.field(default_factory=dict)
    fingerprints: dict = dataclasses.field(default_factory=dict)
    # The sampler's rows for the timed cell, column -> values, in the shape the
    # live review sends as raw_perf. Empty unless the runner was asked to keep
    # them: a long cell sampled every 50 ms is a lot to hold per replay.
    perf_series: dict = dataclasses.field(default_factory=dict)
    error: str = ""
    # Set when the replay died in a prefix cell rather than in the code it was
    # asked to time, so the failure is reported against the notebook rather than
    # against a suggestion that never ran.
    prefix_cell: int | None = None

    @property
    def ok(self) -> bool:
        return self.status == OK


@dataclasses.dataclass
class AttemptRecord:
    """One version of a suggestion that was tried, and what became of it.

    ``error`` is the text the repair was asked to fix, so a history reads as
    the conversation the repair loop actually had. A measured attempt carries
    its median duration and correctness verdict.
    """
    attempt: int
    code: str
    status: str
    error: str = ""
    duration_s: float | None = None
    correctness: str = ""


@dataclasses.dataclass
class BenchmarkResult:
    """What the benchmark concluded about one suggestion (or the baseline).

    ``duration_s`` and ``metrics`` are medians over the timed runs; the first
    run is dropped as warm-up. ``speedup`` is relative to the baseline, and is
    None for the baseline itself.
    """
    label: str
    status: str
    attempts: int = 1
    duration_s: float | None = None
    metrics: dict = dataclasses.field(default_factory=dict)
    speedup: float | None = None
    correctness: str = ""
    differing_names: list[str] = dataclasses.field(default_factory=list)
    # Names the original cell bound and this variant no longer does. Reported
    # separately from a divergence: it says the rewrite stopped producing
    # something, not that it produced it wrongly.
    missing_names: list[str] = dataclasses.field(default_factory=list)
    error: str = ""
    # What the medians above were taken from: every timed replay in order, the
    # warm-up first, so the spread within one variant stays visible.
    run_durations_s: list[float] = dataclasses.field(default_factory=list)
    # Wall-clock of each whole replay, prefix and process start included.
    run_wall_s: list[float | None] = dataclasses.field(default_factory=list)
    # Per replay, the timed cell's perf samples (see ``RunOutcome.perf_series``);
    # empty unless ``ai.benchmark.keep_perf_series`` is on.
    run_perf_series: list[dict] = dataclasses.field(default_factory=list)
    # The result signatures of the last replay - the ones the verdict compared.
    fingerprints: dict = dataclasses.field(default_factory=dict)
    # Every version tried: the model's own, then each repair.
    attempt_history: list[AttemptRecord] = dataclasses.field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == OK
