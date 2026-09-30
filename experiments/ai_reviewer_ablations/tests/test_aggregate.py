"""The summary table: point estimates, and the distance from the baseline.

These cover the path an audit found untested while the report drew its main
conclusion from it.
"""

from __future__ import annotations

import pandas as pd

from jumper_ablations.aggregate.summary import summarise
from jumper_ablations.config.schema import (
    BootstrapConfig,
    ReportingConfig,
    SamplingProtocol,
)
from jumper_ablations.metrics import build_cell_contexts
from jumper_ablations.metrics.base import (
    CATEGORY_ANALYSIS,
    CATEGORY_SUGGESTIONS,
    METHOD_DETERMINISTIC,
    METHOD_JUDGE,
    SCOPE_CELL,
    SCOPE_RUN,
    MetricRow,
)
from jumper_ablations.metrics.registry import get_metric
from tests.factories import make_record

REPORTING = ReportingConfig(
    baseline_ablation="base",
    bootstrap=BootstrapConfig(samples=2000, confidence=0.95, seed=7),
)


def _row(ablation: str, value: float, generation: int, repetition: int = 0):
    unit = (
        f"synthetic-loop__{ablation}__r{repetition:02d}"
        f"__g{generation:02d}__review__none"
    )
    return MetricRow(
        metric="bottleneck_identification",
        reported_value="total_score",
        value=value,
        usecase="synthetic/loop",
        ablation=ablation,
        unit_id=unit,
        scope=SCOPE_RUN,
        category=CATEGORY_ANALYSIS,
        evaluation_method=METHOD_JUDGE,
    )


def _summarise(rows):
    return summarise(
        rows,
        {"bottleneck_identification": get_metric("bottleneck_identification")},
        [],
        REPORTING,
    )


def _only(frame, ablation: str):
    match = frame[frame["ablation"] == ablation]
    assert len(match) == 1
    return match.iloc[0]


def test_the_paired_interval_is_the_interval_of_the_difference():
    # Both presets wander over a wide range, but every pair differs by
    # almost exactly 1.0. The spread of either preset on its own is large;
    # the spread of the difference is tiny. The report used to draw the
    # former around the latter, which is the whole point of this test.
    baseline = [1.0, 5.0, 2.0, 6.0, 3.0, 7.0]
    rows = []
    for generation, value in enumerate(baseline, start=1):
        rows.append(_row("base", value, generation))
        rows.append(_row("no_timing", value + 1.0, generation))

    frame = _summarise(rows)
    preset = _only(frame, "no_timing")

    assert abs(preset["paired_delta"] - 1.0) < 1e-9
    assert preset["paired_n"] == 6

    paired_width = (
        preset["paired_delta_ci_high"] - preset["paired_delta_ci_low"]
    )
    marginal_width = preset["ci_high"] - preset["ci_low"]
    assert paired_width < 1e-6
    assert marginal_width > 1.0
    # The bug was drawing marginal_width around paired_delta: an interval
    # comfortably containing zero for a difference that never varies.
    assert preset["paired_delta_ci_low"] > 0.0


def test_a_difference_of_zero_keeps_an_interval_that_covers_zero():
    rows = []
    for generation, value in enumerate([2.0, 3.0, 4.0, 5.0], start=1):
        rows.append(_row("base", value, generation))
        rows.append(_row("no_perf", value, generation))

    preset = _only(_summarise(rows), "no_perf")

    assert preset["paired_delta"] == 0.0
    assert (
        preset["paired_delta_ci_low"]
        <= 0.0
        <= (preset["paired_delta_ci_high"])
    )


def test_a_single_pair_gets_no_interval_rather_than_a_narrow_one():
    rows = [_row("base", 2.0, 1), _row("no_perf", 3.0, 1)]

    preset = _only(_summarise(rows), "no_perf")

    assert preset["paired_delta"] == 1.0
    assert preset["paired_n"] == 1
    assert pd.isna(preset["paired_delta_ci_low"])
    assert pd.isna(preset["paired_delta_ci_high"])


def test_nothing_to_pair_is_reported_as_nothing():
    # Disjoint generations: a difference of means would still produce a
    # number here, and it would not be a paired one.
    rows = [_row("base", 2.0, 1), _row("no_perf", 9.0, 4)]

    preset = _only(_summarise(rows), "no_perf")

    assert pd.isna(preset["paired_delta"])
    assert preset["paired_n"] == 0


def test_a_second_repetition_does_not_overwrite_the_first():
    # Both repetitions number their generations from 1. Keyed on the
    # generation alone, repetition 1 replaced repetition 0 and half the
    # measurements left the comparison silently.
    rows = []
    for repetition, offset in ((0, 0.0), (1, 4.0)):
        for generation, value in enumerate([1.0, 2.0], start=1):
            rows.append(_row("base", value + offset, generation, repetition))
            rows.append(
                _row("no_tags", value + offset + 0.5, generation, repetition)
            )

    preset = _only(_summarise(rows), "no_tags")

    assert preset["paired_n"] == 4
    assert abs(preset["paired_delta"] - 0.5) < 1e-9


def test_repetitions_are_not_asked_to_redraw_the_same_samples():
    sampling = SamplingProtocol(seed_base=1000)

    first = [sampling.seed_for(g, 0) for g in range(1, 6)]
    second = [sampling.seed_for(g, 1) for g in range(1, 6)]

    assert not set(first) & set(second)
    # Presets still pair: the seed depends on the unit, never on the preset.
    assert sampling.seed_for(3, 0) == sampling.seed_for(3, 0)


def test_an_unset_seed_base_stays_unset():
    sampling = SamplingProtocol(seed_base=None)

    assert sampling.seed_for(3, 1) is None
    assert sampling.as_applied(3, 1)["seed"] is None


class _UnitMeanMetric:
    """A cell-scope metric whose value is controlled per unit.

    Cell-scope metrics reduce a whole cell to one number, so the paired
    difference cannot be taken value by value - both presets have to be
    recomputed on the same resampled units. That machinery is what these
    tests are about, so the metric itself is a stub with numbers the test
    chooses rather than a real one with its own data requirements.
    """

    id = "unit_mean"

    class spec:
        id = "unit_mean"
        scope = SCOPE_CELL
        category = CATEGORY_SUGGESTIONS
        evaluation_method = METHOD_DETERMINISTIC
        reported_values = ("mean",)

    def __init__(self, by_cell: dict):
        self.by_cell = by_cell

    def compute(self, context, parameters):
        values = [
            self.by_cell[(context.ablation_id, unit)]
            for unit in [
                record.identity.unit_key for record in context.records
            ]
        ]
        return {"mean": sum(values) / len(values) if values else None}


def _cell_rows(ablation: str, units):
    return [
        MetricRow(
            metric="unit_mean",
            reported_value="mean",
            value=None,
            usecase="synthetic/loop",
            ablation=ablation,
            unit_id=f"synthetic-loop__{ablation}__r00__g{unit[1]:02d}",
            scope=SCOPE_CELL,
            category=CATEGORY_SUGGESTIONS,
            evaluation_method=METHOD_DETERMINISTIC,
        )
        for unit in units
    ]


def test_a_cell_scope_metric_gets_a_paired_difference_too():
    # Every deterministic metric in this experiment is cell-scope. Before
    # the fix none of them had a paired delta at all: the report compared
    # two point estimates and drew one preset's own spread around the gap.
    units = [(0, generation) for generation in range(1, 7)]
    noisy = [1.0, 6.0, 2.0, 7.0, 3.0, 8.0]
    by_cell = {}
    for unit, value in zip(units, noisy):
        by_cell[("base", unit)] = value
        by_cell[("no_perf", unit)] = value - 2.0

    records = [make_record(generation=unit[1]) for unit in units]
    contexts = build_cell_contexts(
        records
        + [
            make_record(
                usecase="synthetic/loop",
                ablation="no_perf",
                generation=unit[1],
            )
            for unit in units
        ],
        {},
    )
    metric = _UnitMeanMetric(by_cell)

    frame = summarise(
        _cell_rows("base", units) + _cell_rows("no_perf", units),
        {"unit_mean": metric},
        contexts,
        REPORTING,
    )
    preset = _only(frame, "no_perf")

    assert abs(preset["paired_delta"] + 2.0) < 1e-9
    assert preset["paired_n"] == 6
    # The difference is exactly -2.0 on every resample, so its interval is a
    # point - while either preset on its own spans 1..8.
    width = preset["paired_delta_ci_high"] - preset["paired_delta_ci_low"]
    assert width < 1e-6
    assert preset["paired_delta_ci_high"] < 0.0


def test_a_cell_scope_difference_needs_units_in_both_presets():
    base_units = [(0, 1), (0, 2)]
    other_units = [(0, 3), (0, 4)]
    by_cell = {("base", u): 1.0 for u in base_units}
    by_cell.update({("no_perf", u): 5.0 for u in other_units})

    contexts = build_cell_contexts(
        [make_record(generation=u[1]) for u in base_units]
        + [
            make_record(ablation="no_perf", generation=u[1])
            for u in other_units
        ],
        {},
    )

    frame = summarise(
        _cell_rows("base", base_units) + _cell_rows("no_perf", other_units),
        {"unit_mean": _UnitMeanMetric(by_cell)},
        contexts,
        REPORTING,
    )
    preset = _only(frame, "no_perf")

    assert pd.isna(preset["paired_delta"])
    assert preset["paired_n"] == 0
