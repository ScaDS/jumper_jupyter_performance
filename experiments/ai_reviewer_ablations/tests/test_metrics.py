"""The computed metrics, against records whose answers are known by hand."""

from __future__ import annotations

import pytest

from jumper_ablations.metrics import build_cell_contexts, get_metric
from tests.factories import make_cell


@pytest.fixture
def cell():
    return build_cell_contexts(make_cell(generations=3))[0]


def _values(metric_id: str, context, parameters: dict | None = None) -> dict:
    return get_metric(metric_id).compute(context, parameters or {})


def test_correctness_rates_never_count_unverified_as_correct(cell):
    values = _values("correctness_rates", cell)

    # Per generation: one match, one differs, one that never ran.
    assert values["suggestions"] == 9
    assert values["verified_rate"] == pytest.approx(1 / 3)
    assert values["differs_rate"] == pytest.approx(1 / 3)
    assert values["failed_rate"] == pytest.approx(1 / 3)


def test_best_speedup_separates_fastest_from_fastest_correct(cell):
    values = _values("best_speedup", cell)

    # The 8x option is the fastest, and it computed something else.
    assert values["mean_best_speedup"] == pytest.approx(8.0)
    assert values["mean_best_correct_speedup"] == pytest.approx(4.0)


def test_correctness_conditioned_speedup_reports_its_coverage(cell):
    values = _values("correctness_conditioned_speedup", cell)

    assert values["geometric_mean_speedup"] == pytest.approx(4.0)
    assert values["coverage"] == pytest.approx(1 / 3)
    assert values["suggestions"] == 9


def test_pass_at_k_requires_correctness_not_just_execution(cell):
    values = _values("pass_at_k", cell)

    # The first-ranked option is the correct one in every generation.
    assert values["pass_at_1"] == pytest.approx(1.0)
    assert values["pass_at_k"] == pytest.approx(1.0)
    assert values["mean_k"] == pytest.approx(3.0)


def test_execution_success_separates_repaired_from_first_try(cell):
    values = _values("execution_success", cell)

    # Two of three ran, neither needed a repair to do so.
    assert values["post_repair_success_rate"] == pytest.approx(2 / 3)
    assert values["pre_repair_success_rate"] == pytest.approx(2 / 3)
    assert values["repair_uplift"] == pytest.approx(0.0)


def test_requested_vs_returned_reads_the_configured_expectation(cell):
    values = _values("requested_vs_returned", cell, {"requested": 5})

    assert values["requested"] == 5
    assert values["mean_returned"] == pytest.approx(3.0)
    assert values["mean_shortfall"] == pytest.approx(2.0)
    assert values["compliance_rate"] == pytest.approx(0.0)


def test_rank_quality_only_ranks_correct_suggestions(cell):
    values = _values("rank_quality", cell)

    # Only one correct option per generation, so it is trivially the fastest
    # and there is nothing to correlate.
    assert values["first_is_fastest_rate"] == pytest.approx(1.0)
    assert values["comparable_generations"] == 0
    assert values["mean_rank_correlation"] is None


def test_replay_mode_agreement_is_undefined_for_a_single_mode(cell):
    values = _values("replay_mode_agreement", cell)

    assert values["modes_compared"] == 1
    assert values["fallback_rate"] == pytest.approx(0.0)
    assert "correctness_agreement" not in values


def test_end_to_end_cost_reads_the_recorded_calls(cell):
    values = _values("end_to_end_cost", cell)

    assert values["mean_review_latency_s"] == pytest.approx(5.0)
    assert values["mean_total_tokens"] == pytest.approx(430.0)
    assert values["mean_benchmark_wall_s"] == pytest.approx(12.0)
