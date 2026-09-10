"""The record: what one invocation of the reviewer produced.

This is the contract between the expensive half of the experiment and the
cheap half. Running the reviewer costs a model call and a benchmark that
replays a whole notebook prefix per measurement; scoring it costs a loop. They
are separate commands so that adding a metric re-reads records instead of
re-running the pipeline, and this file is what makes that possible: everything
a metric could want is written down once, at capture time, inside the kernel
that produced it.

Nothing here is derived. Speedups, correctness verdicts and repair counts are
copied out of the reviewer's own `BenchmarkResult`; the messages are the
verbatim ones the model received. Interpretation belongs to the metrics.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

# What `phase` can say about how a record came to be.
PHASE_REVIEW = "review"
PHASE_REBENCHMARK = "rebenchmark"

# The reviewer's own baseline key inside `benchmarks`.
BASELINE_LABEL = "baseline"


class SuggestionRecord(BaseModel):
    """One proposed rewrite, as the reviewer finally held it.

    ``code`` is post-repair: the benchmark writes the version it actually
    measured back onto the suggestion, so this is what the numbers describe.
    """

    index: int
    title: str
    description: str
    code: str
    target_cell_index: int | None = None


class BenchmarkRecord(BaseModel):
    """What the benchmark concluded about one suggestion, or the baseline."""

    label: str
    status: str
    attempts: int = 1
    duration_s: float | None = None
    metrics: dict = Field(default_factory=dict)
    speedup: float | None = None
    # The reviewer's fingerprint verdict: match / differs / unverified.
    correctness: str = ""
    differing_names: list[str] = Field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class LLMCall(BaseModel):
    """One model call the reviewer made while producing this record."""

    step: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    model: str = ""


class RunIdentity(BaseModel):
    """Which cell of the experiment grid this record belongs to.

    ``generation`` is the paired index: generation 3 of `no_timing` and
    generation 3 of `base` were asked for under the same conditions, which is
    what lets the report compare presets pairwise instead of in bulk.
    """

    record_id: str
    usecase: str
    ablation: str
    ablation_family: str = "context"
    repetition: int = 0
    generation: int = 0
    phase: str = PHASE_REVIEW
    # The reviewer's own 8-hex id, so a --resume benchmark record can be tied
    # back to the review that produced the suggestions.
    reviewer_run_id: str = ""
    requested_replay_mode: str = ""
    created_at: str = ""
    suite: str = ""
    run_id: str = ""


class RunInputs(BaseModel):
    """Everything the model was given, exactly as it was given.

    ``analyze_messages`` and ``suggest_messages`` are the verbatim message
    lists; a judge reads those rather than a reconstruction, which is the only
    way the judged sources can be claimed to be the reviewer's sources.
    """

    overrides: dict = Field(default_factory=dict)
    enabled_sources: dict = Field(default_factory=dict)
    context_payload: dict = Field(default_factory=dict)
    analyze_messages: list[dict] = Field(default_factory=list)
    suggest_messages: list[dict] = Field(default_factory=list)
    cell_code: str = ""
    level: str = "process"
    note: str = ""


class RunOutputs(BaseModel):
    analysis: str = ""
    analysis_reasoning: str = ""
    suggestions: list[SuggestionRecord] = Field(default_factory=list)
    benchmarks: dict[str, BenchmarkRecord] = Field(default_factory=dict)

    def baseline(self) -> BenchmarkRecord | None:
        return self.benchmarks.get(BASELINE_LABEL)

    def verdicts(self) -> dict[int, BenchmarkRecord]:
        """Benchmark rows keyed by suggestion index, baseline excluded."""
        return {
            int(label): record
            for label, record in self.benchmarks.items()
            if label != BASELINE_LABEL and label.isdigit()
        }


class RunCost(BaseModel):
    llm_calls: list[LLMCall] = Field(default_factory=list)
    llm_latency_s: float = 0.0
    total_tokens: int | None = None
    # Wall time of the magic itself, benchmark included.
    command_wall_s: float = 0.0


class RunEnvironment(BaseModel):
    """What the numbers were produced on, and whether they can be trusted.

    ``actual_replay_mode`` matters more than it looks: the reviewer degrades a
    fast replay mode to `full` with a warning rather than failing, so without
    recording what ran, an experiment can report two modes that were secretly
    one.
    """

    llm_config: dict = Field(default_factory=dict)
    hardware: dict = Field(default_factory=dict)
    actual_replay_mode: str = ""
    degraded: bool = False
    warnings: list[str] = Field(default_factory=list)
    package_versions: dict = Field(default_factory=dict)


class RunRecord(BaseModel):
    """One reviewer invocation, complete."""

    identity: RunIdentity
    inputs: RunInputs = Field(default_factory=RunInputs)
    outputs: RunOutputs = Field(default_factory=RunOutputs)
    cost: RunCost = Field(default_factory=RunCost)
    environment: RunEnvironment = Field(default_factory=RunEnvironment)

    @property
    def cell_key(self) -> tuple[str, str]:
        """The grid cell this record belongs to: (usecase, ablation)."""
        return (self.identity.usecase, self.identity.ablation)
