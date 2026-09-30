"""What a pass owes, and what does not count towards it.

A pass that recorded less than the protocol asked for has measured a
different experiment from the one the report will describe, and a generation
the reviewer answered with no context has measured nothing at all.
"""

from __future__ import annotations

from jumper_ablations.config.schema import BenchmarkProtocol, ProtocolConfig
from jumper_ablations.metrics.context import drop_empty_context
from jumper_ablations.records.schema import PHASE_REBENCHMARK, PHASE_REVIEW
from jumper_ablations.runner.executor import PassOutcome, _expected_captures
from jumper_ablations.runtime.session import EMPTY_CONTEXT_WARNING
from jumper_ablations.usecases.registry import (
    Usecase,
    UsecaseBenchmark,
    UsecaseManifest,
)
from tests.factories import make_record


def _usecase(replay_mode: str = "full", extra: list | None = None) -> Usecase:
    return Usecase(
        manifest=UsecaseManifest(
            id="synthetic/loop",
            benchmark=UsecaseBenchmark(
                replay_mode=replay_mode,
                extra_replay_modes=extra or [],
            ),
        )
    )


def test_without_a_benchmark_a_generation_owes_only_its_review():
    protocol = ProtocolConfig(benchmark=BenchmarkProtocol(enabled=False))

    assert _expected_captures(protocol, _usecase(), inline=False) == 1


def test_the_two_command_shape_owes_a_record_per_mode():
    protocol = ProtocolConfig(
        benchmark=BenchmarkProtocol(extra_replay_modes=["fork"])
    )

    # The review, the mode the usecase asked for, and the extra one.
    assert _expected_captures(protocol, _usecase(), inline=False) == 3


def test_the_usecase_and_the_protocol_modes_are_unioned_without_repeats():
    protocol = ProtocolConfig(
        benchmark=BenchmarkProtocol(extra_replay_modes=["fork", "full"])
    )
    usecase = _usecase(replay_mode="full", extra=["fork", "dill"])

    # full, fork, dill - named twice between the two sources, owed once.
    assert _expected_captures(protocol, usecase, inline=False) == 4


def test_the_inline_shape_does_not_owe_a_separate_first_measurement():
    protocol = ProtocolConfig(benchmark=BenchmarkProtocol())

    assert _expected_captures(protocol, _usecase(), inline=True) == 1


def test_a_pass_short_of_its_records_is_not_complete():
    outcome = PassOutcome(
        usecase="synthetic/loop",
        ablation="base",
        repetition=0,
        status="ok",
        captures=8,
        expected_captures=10,
    )

    assert not outcome.complete


def test_a_pass_with_an_empty_context_generation_is_not_complete():
    outcome = PassOutcome(
        usecase="synthetic/loop",
        ablation="base",
        repetition=0,
        status="ok",
        captures=10,
        expected_captures=10,
        empty_context=1,
    )

    assert not outcome.complete


def _empty_context(record):
    record.environment.warnings.append(EMPTY_CONTEXT_WARNING)
    return record


def test_an_empty_context_generation_leaves_with_its_benchmark():
    # The benchmark timed real code, but the suggestions it timed came out
    # of an empty prompt: keeping it puts the same confabulation into the
    # speedup numbers by another route.
    good_review = make_record(generation=1)
    good_bench = make_record(generation=1, phase=PHASE_REBENCHMARK)
    bad_review = _empty_context(make_record(generation=2))
    bad_bench = make_record(generation=2, phase=PHASE_REBENCHMARK)

    kept = drop_empty_context([good_review, good_bench, bad_review, bad_bench])

    assert {record.identity.record_id for record in kept} == {
        good_review.identity.record_id,
        good_bench.identity.record_id,
    }


def test_records_are_left_alone_when_every_reviewer_saw_something():
    records = [
        make_record(generation=1),
        make_record(generation=1, phase=PHASE_REBENCHMARK),
    ]

    assert drop_empty_context(records) == records


def test_an_empty_context_generation_never_reaches_a_metric():
    from jumper_ablations.metrics import build_run_contexts

    contexts = build_run_contexts(
        [make_record(generation=1), _empty_context(make_record(generation=2))]
    )

    assert [context.record.identity.generation for context in contexts] == [1]
    assert all(
        context.record.identity.phase == PHASE_REVIEW for context in contexts
    )


def test_each_pass_gets_its_own_scratch(tmp_path):
    # The working directory is shared by every pass of a usecase family on
    # purpose - cell_77 reads the store cell_40's pipeline wrote - so the
    # scratch that dask and the benchmark write must not be shared with it.
    # Two passes at once otherwise build their clusters on top of each other.
    from jumper_ablations.runner.executor import _kernel_environment

    first = _kernel_environment(tmp_path / "usecase" / "base" / "r0")
    second = _kernel_environment(tmp_path / "usecase" / "no_perf" / "r0")

    for variable in ("TMPDIR", "DASK_TEMPORARY_DIRECTORY"):
        assert first[variable] != second[variable]
        assert first[variable].endswith("tmp")
