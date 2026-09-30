"""Synthetic records, so the scoring half can be tested without the model.

Producing a real record costs a model call and a benchmark that replays a
notebook prefix many times. Everything downstream of that - the metrics, the
judge round trip, the aggregation - only reads records, so it is testable
against records built here in microseconds.
"""

from __future__ import annotations

from jumper_ablations.records.schema import (
    PHASE_REBENCHMARK,
    PHASE_REVIEW,
    BenchmarkRecord,
    LLMCall,
    RunCost,
    RunEnvironment,
    RunIdentity,
    RunInputs,
    RunOutputs,
    RunRecord,
    SuggestionRecord,
)

ALL_SOURCES = (
    "code",
    "timing",
    "tags",
    "perf",
    "raw_perf",
    "hardware",
    "packages",
)


def make_inputs(disabled: tuple[str, ...] = ()) -> RunInputs:
    enabled = {source: source not in disabled for source in ALL_SOURCES}
    payload = {
        source: ({"value": 1} if enabled[source] else {})
        for source in ALL_SOURCES
    }
    return RunInputs(
        overrides={source: False for source in disabled},
        enabled_sources=enabled,
        context_payload=payload,
        analyze_messages=[
            {"role": "SystemMessage", "content": "system"},
            {"role": "HumanMessage", "content": "human"},
        ],
        suggest_messages=[{"role": "SystemMessage", "content": "system"}],
        cell_code="total = 0",
    )


def make_suggestions(count: int = 3) -> list[SuggestionRecord]:
    return [
        SuggestionRecord(
            index=index,
            title=f"option {index}",
            description=f"description {index}",
            code=f"total = {index}",
        )
        for index in range(1, count + 1)
    ]


def make_benchmarks(verdicts: list[dict]) -> dict[str, BenchmarkRecord]:
    """``verdicts`` is one dict per suggestion, plus an implicit baseline."""
    built = {
        "baseline": BenchmarkRecord(
            label="baseline",
            status="ok",
            duration_s=10.0,
            correctness="match",
            metrics={"cpu_util_mean": 0.9, "gpu_util_mean": 0.0},
        )
    }
    for index, verdict in enumerate(verdicts, 1):
        built[str(index)] = BenchmarkRecord(
            label=str(index),
            status=verdict.get("status", "ok"),
            attempts=verdict.get("attempts", 1),
            duration_s=verdict.get("duration_s", 5.0),
            speedup=verdict.get("speedup", 2.0),
            correctness=verdict.get("correctness", "match"),
            metrics=verdict.get(
                "metrics",
                {"cpu_util_mean": 0.4, "gpu_util_mean": 0.0},
            ),
            error=verdict.get("error", ""),
        )
    return built


def make_record(
    usecase: str = "synthetic/loop",
    ablation: str = "base",
    generation: int = 1,
    phase: str = PHASE_REVIEW,
    disabled: tuple[str, ...] = (),
    suggestions: int = 3,
    verdicts: list[dict] | None = None,
    replay_mode: str = "full",
    degraded: bool = False,
) -> RunRecord:
    identity = RunIdentity(
        record_id=(
            f"{usecase.replace('/', '-')}__{ablation}"
            f"__r00__g{generation:02d}__{phase}__{replay_mode}"
        ),
        usecase=usecase,
        ablation=ablation,
        generation=generation,
        phase=phase,
        reviewer_run_id=f"run{generation:02d}",
        requested_replay_mode=replay_mode,
    )
    outputs = RunOutputs(
        analysis="A sequential loop dominates the runtime.",
        suggestions=make_suggestions(suggestions),
        benchmarks=make_benchmarks(verdicts) if verdicts is not None else {},
    )
    cost = RunCost(
        llm_calls=[
            LLMCall(
                step="analyze",
                latency_s=2.0,
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
            ),
            LLMCall(
                step="suggest",
                latency_s=3.0,
                input_tokens=200,
                output_tokens=80,
                total_tokens=280,
            ),
        ],
        llm_latency_s=5.0,
        total_tokens=430,
        command_wall_s=12.0,
    )
    return RunRecord(
        identity=identity,
        inputs=make_inputs(disabled),
        outputs=outputs,
        cost=cost,
        environment=RunEnvironment(
            actual_replay_mode="full" if degraded else replay_mode,
            degraded=degraded,
        ),
    )


def make_cell(
    usecase: str = "synthetic/loop",
    ablation: str = "base",
    generations: int = 3,
    disabled: tuple[str, ...] = (),
    verdicts: list[dict] | None = None,
) -> list[RunRecord]:
    """A review record plus its benchmark record, per generation.

    This is the two-command shape the runner uses by default: the review is
    captured on its own, the benchmark re-opens it.
    """
    records = []
    for generation in range(1, generations + 1):
        records.append(
            make_record(
                usecase=usecase,
                ablation=ablation,
                generation=generation,
                phase=PHASE_REVIEW,
                disabled=disabled,
                replay_mode="",
            )
        )
        records.append(
            make_record(
                usecase=usecase,
                ablation=ablation,
                generation=generation,
                phase=PHASE_REBENCHMARK,
                disabled=disabled,
                verdicts=(
                    verdicts if verdicts is not None else _default_verdicts()
                ),
            )
        )
    return records


def _default_verdicts() -> list[dict]:
    """One clean win, one that diverged, one that never ran."""
    return [
        {"speedup": 4.0, "correctness": "match"},
        {"speedup": 8.0, "correctness": "differs"},
        {
            "status": "failed",
            "duration_s": None,
            "speedup": None,
            "attempts": 2,
            "error": "NameError",
        },
    ]
