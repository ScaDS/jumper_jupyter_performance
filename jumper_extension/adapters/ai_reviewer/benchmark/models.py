import dataclasses

OK = "ok"
FAILED = "failed"
TIMEOUT = "timeout"


@dataclasses.dataclass
class RunOutcome:
    """One replay of a cell: either it produced numbers, or it did not."""
    status: str
    duration_s: float | None = None
    wall_s: float | None = None
    metrics: dict = dataclasses.field(default_factory=dict)
    fingerprints: dict = dataclasses.field(default_factory=dict)
    error: str = ""
    # Set when the replay died in a prefix cell rather than in the code it was
    # asked to time, so the failure is reported against the notebook rather than
    # against a suggestion that never ran.
    prefix_cell: int | None = None

    @property
    def ok(self) -> bool:
        return self.status == OK


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

    @property
    def ok(self) -> bool:
        return self.status == OK
