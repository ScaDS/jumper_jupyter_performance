"""The KernelBench and ECCO metrics, against verifications known by hand."""

from __future__ import annotations

import pytest

from jumper_ablations.metrics import build_cell_contexts, get_metric
from jumper_ablations.verification.schema import (
    EAGER,
    HUMAN_FAST,
    ORIGINAL,
    TORCH_COMPILE,
    Measurement,
    References,
    UsecaseVerification,
    Verification,
)
from tests.factories import make_cell

USECASE = "synthetic/loop"


def _measurement(
    label: str,
    time_s: float | None,
    memory: int | None = None,
    correct: bool | None = True,
    status: str = "ok",
) -> Measurement:
    return Measurement(
        label=label,
        status=status,
        time_s=time_s,
        peak_memory_bytes=memory,
        correct=correct,
    )


def _cell(references: References, suggestions: list[Measurement]):
    """Three generations, each review verified with the same suggestions."""
    reviews = {
        f"run{generation:02d}": Verification(
            usecase=USECASE,
            ablation="base",
            reviewer_run_id=f"run{generation:02d}",
            suggestions={
                str(index): measurement
                for index, measurement in enumerate(suggestions, 1)
            },
        )
        for generation in (1, 2, 3)
    }
    verification = UsecaseVerification(references=references, reviews=reviews)
    return build_cell_contexts(
        make_cell(generations=3),
        verifications={USECASE: verification},
    )[0]


def _cell_per_generation(references: References, per_generation: dict):
    """Generations that differ from each other, keyed by generation number."""
    reviews = {
        f"run{generation:02d}": Verification(
            usecase=USECASE,
            ablation="base",
            reviewer_run_id=f"run{generation:02d}",
            suggestions={
                str(index): measurement
                for index, measurement in enumerate(measurements, 1)
            },
        )
        for generation, measurements in per_generation.items()
    }
    verification = UsecaseVerification(references=references, reviews=reviews)
    return build_cell_contexts(
        make_cell(generations=len(per_generation)),
        verifications={USECASE: verification},
    )[0]


@pytest.fixture
def kernelbench_cell():
    references = References(
        usecase=USECASE,
        kind="kernelbench",
        entries={
            EAGER: _measurement(EAGER, 2.0),
            TORCH_COMPILE: _measurement(TORCH_COMPILE, 1.0),
        },
    )
    return _cell(
        references,
        [
            # 4x over eager, 2x over compile.
            _measurement("1", 0.5),
            # 1.25x over eager, slower than compile.
            _measurement("2", 1.6),
            # Fast and wrong.
            _measurement("3", 0.1, correct=False),
            _measurement("4", None, correct=None, status="failed"),
        ],
    )


@pytest.fixture
def ecco_cell():
    references = References(
        usecase=USECASE,
        kind="ecco",
        entries={
            ORIGINAL: _measurement(ORIGINAL, 20.0, memory=400),
            HUMAN_FAST: _measurement(HUMAN_FAST, 2.0, memory=100),
        },
        population=[
            _measurement("a", 30.0, memory=500),
            _measurement("b", 10.0, memory=300),
            _measurement("c", 1.0, memory=50),
            # A wrong answer is not on the spectrum; a timeout is, as slowest.
            _measurement("d", 0.5, memory=10, correct=False),
            _measurement("e", None, correct=None, status="timeout"),
        ],
    )
    return _cell(
        references,
        [
            _measurement("1", 4.0, memory=200),
            _measurement("2", 1.0, memory=100, correct=False),
        ],
    )


def _values(metric_id: str, context) -> dict:
    return get_metric(metric_id).compute(context, {})


def test_fast_p_counts_every_suggestion_in_the_denominator(kernelbench_cell):
    values = _values("fast_p", kernelbench_cell)

    assert values["suggestions"] == 12
    assert values["fast_0"] == pytest.approx(2 / 4)
    assert values["fast_1"] == pytest.approx(2 / 4)
    assert values["fast_1_5"] == pytest.approx(1 / 4)
    assert values["fast_2"] == pytest.approx(1 / 4)


def test_allclose_correctness_counts_failures_as_incorrect(kernelbench_cell):
    values = _values("allclose_correctness", kernelbench_cell)

    assert values["allclose_rate"] == pytest.approx(2 / 4)


def test_compile_speedup_is_over_correct_suggestions_only(kernelbench_cell):
    values = _values("compile_speedup", kernelbench_cell)

    assert values["geometric_mean_compile_speedup"] == pytest.approx(
        (2.0 * 1.0 / 1.6) ** 0.5
    )
    assert values["geometric_mean_eager_speedup"] == pytest.approx(
        (4.0 * 1.25) ** 0.5
    )
    assert values["correct_suggestions"] == 6


def test_ecco_reference_ratios(ecco_cell):
    assert _values("memory_reduction", ecco_cell)[
        "geometric_mean_memory_reduction"
    ] == pytest.approx(2.0)
    assert _values("reference_speedup", ecco_cell)[
        "geometric_mean_reference_speedup"
    ] == pytest.approx(0.5)
    assert _values("reference_speedup", ecco_cell)[
        "best_reference_speedup"
    ] == pytest.approx(0.5)
    assert _values("reference_memory_reduction", ecco_cell)[
        "best_reference_memory_reduction"
    ] == pytest.approx(0.5)
    assert _values("reference_memory_reduction", ecco_cell)[
        "geometric_mean_reference_memory_reduction"
    ] == pytest.approx(0.5)


def test_ecco_percentiles_rank_against_correct_submissions(ecco_cell):
    runtime = _values("runtime_percentile", ecco_cell)
    memory = _values("memory_percentile", ecco_cell)

    # Spectrum 30, 10, 1 and the timeout: three of four are slower than 4.0.
    assert runtime["mean_runtime_percentile"] == pytest.approx(75.0)
    # One correct suggestion per generation, so the best of it is the same.
    assert runtime["best_runtime_percentile"] == pytest.approx(75.0)
    assert runtime["submissions"] == 4
    # Memory spectrum 500, 300, 50: two of three use more than 200.
    assert memory["mean_memory_percentile"] == pytest.approx(200 / 3)
    assert memory["best_memory_percentile"] == pytest.approx(200 / 3)
    assert memory["submissions"] == 3


def _percentile_references() -> References:
    # Spectrum 30, 10, 1 and a timeout: 4.0s beats three of four, 20.0s two.
    return References(
        usecase=USECASE,
        kind="ecco",
        entries={HUMAN_FAST: _measurement(HUMAN_FAST, 2.0)},
        population=[
            _measurement("a", 30.0),
            _measurement("b", 10.0),
            _measurement("c", 1.0),
            _measurement("d", None, correct=None, status="timeout"),
        ],
    )


def test_a_generation_is_reduced_to_its_best_correct_suggestion():
    # Both values are about the option a user takes, so the mediocre option of
    # a generation never drags its number down - only the other generations do.
    cell = _cell(
        _percentile_references(),
        [
            _measurement("1", 4.0),   # 75%
            _measurement("2", 20.0),  # 50%, same generation
        ],
    )

    values = _values("runtime_percentile", cell)

    assert values["mean_runtime_percentile"] == pytest.approx(75.0)
    assert values["best_runtime_percentile"] == pytest.approx(75.0)
    assert values["correct_suggestions"] == 6  # three generations, two each


def test_the_mean_averages_generations_and_the_best_does_not():
    # One lucky generation and one ordinary one. The mean says what a
    # generation achieves typically; the best says what was ever achieved.
    cell = _cell_per_generation(
        _percentile_references(),
        {
            1: [_measurement("1", 4.0)],   # 75%
            2: [_measurement("1", 20.0)],  # 50%
        },
    )

    values = _values("runtime_percentile", cell)

    assert values["mean_runtime_percentile"] == pytest.approx((75.0 + 50.0) / 2)
    assert values["best_runtime_percentile"] == pytest.approx(75.0)


def test_a_generation_without_a_correct_suggestion_lowers_neither_value():
    # Left out rather than counted as 0%: coverage is where that failure is
    # reported, and folding it in here would mix two different ones.
    cell = _cell_per_generation(
        _percentile_references(),
        {
            1: [_measurement("1", 4.0)],
            2: [_measurement("1", 0.5, correct=False)],
        },
    )

    values = _values("runtime_percentile", cell)

    assert values["mean_runtime_percentile"] == pytest.approx(75.0)
    assert values["best_runtime_percentile"] == pytest.approx(75.0)


def test_memory_percentile_splits_the_same_way():
    references = References(
        usecase=USECASE,
        kind="ecco",
        entries={HUMAN_FAST: _measurement(HUMAN_FAST, 2.0, memory=100)},
        # Spectrum 500, 300, 50: 200 bytes beat two of three, 400 only one.
        population=[
            _measurement("a", 30.0, memory=500),
            _measurement("b", 10.0, memory=300),
            _measurement("c", 1.0, memory=50),
        ],
    )
    cell = _cell_per_generation(
        references,
        {
            1: [_measurement("1", 4.0, memory=200)],
            2: [_measurement("1", 4.0, memory=400)],
        },
    )

    values = _values("memory_percentile", cell)

    assert values["mean_memory_percentile"] == pytest.approx((200 / 3 + 100 / 3) / 2)
    assert values["best_memory_percentile"] == pytest.approx(200 / 3)


def test_best_reference_speedup_is_the_fastest_draw_of_the_whole_cell():
    # Human fast at 2.0s. One generation got to 0.5s - four times the human -
    # another only to 4.0s. The geometric mean sits between them; the best
    # reports the one draw that beat the person.
    references = References(
        usecase=USECASE,
        kind="ecco",
        entries={HUMAN_FAST: _measurement(HUMAN_FAST, 2.0)},
    )
    cell = _cell_per_generation(
        references,
        {
            1: [_measurement("1", 0.5)],   # 4.0x the human
            2: [_measurement("1", 4.0)],   # 0.5x
        },
    )

    values = _values("reference_speedup", cell)

    assert values["geometric_mean_reference_speedup"] == pytest.approx(
        (4.0 * 0.5) ** 0.5
    )
    assert values["best_reference_speedup"] == pytest.approx(4.0)


def test_best_reference_memory_reduction_is_the_leanest_draw():
    references = References(
        usecase=USECASE,
        kind="ecco",
        entries={HUMAN_FAST: _measurement(HUMAN_FAST, 2.0, memory=400)},
    )
    cell = _cell_per_generation(
        references,
        {
            1: [_measurement("1", 4.0, memory=100)],  # 4x leaner than the human
            2: [_measurement("1", 4.0, memory=800)],  # 0.5x
        },
    )

    values = _values("reference_memory_reduction", cell)

    assert values["geometric_mean_reference_memory_reduction"] == pytest.approx(
        (4.0 * 0.5) ** 0.5
    )
    assert values["best_reference_memory_reduction"] == pytest.approx(4.0)


def test_both_values_abstain_when_nothing_was_correct():
    references = References(
        usecase=USECASE,
        kind="ecco",
        entries={HUMAN_FAST: _measurement(HUMAN_FAST, 2.0)},
        population=[_measurement("a", 30.0), _measurement("b", 1.0)],
    )
    cell = _cell(references, [_measurement("1", 20.0, correct=False)])

    values = _values("runtime_percentile", cell)

    assert values["best_runtime_percentile"] is None
    assert values["mean_runtime_percentile"] is None
    assert values["correct_suggestions"] == 0


def test_paper_metrics_abstain_without_verification():
    context = build_cell_contexts(make_cell(generations=1))[0]

    assert _values("fast_p", context)["fast_1"] is None
    assert _values("runtime_percentile", context)["mean_runtime_percentile"] is None
    assert _values("runtime_percentile", context)["best_runtime_percentile"] is None
    assert _values("memory_percentile", context)["best_memory_percentile"] is None
    assert _values("reference_speedup", context)["best_reference_speedup"] is None
    assert _values("reference_memory_reduction", context)[
        "best_reference_memory_reduction"
    ] is None
    assert _values("compile_speedup", context)["correct_suggestions"] == 0
