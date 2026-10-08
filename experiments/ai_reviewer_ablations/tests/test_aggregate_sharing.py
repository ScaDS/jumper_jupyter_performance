"""The report may be fast, but not by changing any number it reports.

Sharing resamples between a metric's reported values, and between the
interval and the paired difference, is an exact restructuring: a metric
answers for all of its reported values in one call, and a draw depends only
on the unit set it is drawn from. Both of those are easy to break while the
report still produces plausible output, so they are asserted here rather
than left to a reviewer's eye.
"""

from __future__ import annotations

import dataclasses

import pandas as pd

from jumper_ablations.aggregate.summary import CellBootstrap, summarise
from jumper_ablations.config.schema import BootstrapConfig, ReportingConfig
from jumper_ablations.metrics.base import (
    CATEGORY_SUGGESTIONS,
    KIND_RESULT,
    METHOD_DETERMINISTIC,
    SCOPE_CELL,
    MetricRow,
    MetricSpec,
)
from jumper_ablations.metrics.context import CellContext
from jumper_ablations.records.schema import PHASE_REVIEW


@dataclasses.dataclass(frozen=True)
class _Identity:
    unit_key: tuple[int, int]
    generation: int
    phase: str = PHASE_REVIEW


@dataclasses.dataclass(frozen=True)
class _Record:
    identity: _Identity


def _cell(ablation: str, units) -> CellContext:
    return CellContext(
        usecase_id="a/one",
        ablation_id=ablation,
        records=tuple(
            _Record(_Identity(unit, unit[1])) for unit in units
        ),
    )


class _Counted:
    """A cell metric that reports three values and counts its calls."""

    spec = MetricSpec(
        id="counted",
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=("first", "second", "third"),
        description="Three numbers derived from the units in the cell.",
        supporting_values=("third",),
    )

    def __init__(self):
        self.calls = 0

    @property
    def id(self) -> str:
        return self.spec.id

    def compute(self, context, parameters) -> dict:
        self.calls += 1
        units = [record.identity.unit_key for record in context.records]
        total = sum(unit[1] for unit in units)
        return {
            "first": float(total),
            "second": float(total) / max(len(units), 1),
            "third": float(len(units)),
        }

    def rows(self, context, values: dict, note: str = "") -> list[MetricRow]:
        return [
            MetricRow(
                metric=self.spec.id,
                reported_value=name,
                value_kind=self.spec.kind_of(name),
                value=values.get(name),
                usecase=context.usecase_id,
                ablation=context.ablation_id,
                unit_id=context.unit_id,
                scope=SCOPE_CELL,
                category=CATEGORY_SUGGESTIONS,
                evaluation_method=METHOD_DETERMINISTIC,
                sample_size=context.sample_size,
                note=note,
            )
            for name in self.spec.reported_values
        ]


def _reporting(**overrides) -> ReportingConfig:
    return ReportingConfig(
        baseline_ablation="base",
        bootstrap=BootstrapConfig(samples=100, confidence=0.95, seed=0),
        figures=False,
        **overrides,
    )


UNITS = ((0, 1), (0, 2), (0, 3), (0, 4))


def _rows(metric, cells) -> list[MetricRow]:
    out = []
    for cell in cells:
        out.extend(metric.rows(cell, metric.compute(cell, {})))
    return out


def test_a_metric_is_computed_once_per_resample_not_once_per_value():
    # Three reported values out of one call. Computing per value would
    # triple the work and change nothing, which is exactly the kind of
    # waste that is invisible in the output.
    metric = _Counted()
    cells = [_cell("base", UNITS), _cell("no_code", UNITS)]
    rows = _rows(metric, cells)
    before = metric.calls

    summarise(
        rows=rows,
        metrics_by_id={metric.id: metric},
        cell_contexts=cells,
        reporting=_reporting(workers=1),
    )
    draws = min(100, 400)
    spent = metric.calls - before

    # Two presets' resamples, plus one point estimate each on the shared
    # units. Three reported values add nothing.
    assert spent == draws * 2 + 2, spent


def test_the_interval_and_the_paired_difference_share_their_draws():
    # Both resample the same units of the same cell, so drawing twice would
    # be two different samples of one population and twice the cost.
    metric = _Counted()
    cells = [_cell("base", UNITS), _cell("no_code", UNITS)]

    summarise(
        rows=_rows(metric, cells),
        metrics_by_id={metric.id: metric},
        cell_contexts=cells,
        reporting=_reporting(workers=1),
    )

    bootstrap = CellBootstrap(_reporting(), {})
    units = bootstrap.units_of(cells[1])
    shared = bootstrap.shared_units(cells[1], cells[0])
    assert units == shared
    assert bootstrap.drawn(units) is bootstrap.drawn(shared)


def test_presets_that_ran_different_units_do_not_share_draws():
    # The saving is only exact while the populations coincide. When they do
    # not, reusing a draw would resample the wrong set.
    bootstrap = CellBootstrap(_reporting(), {})
    mine = _cell("no_code", UNITS)
    baseline = _cell("base", UNITS[:2])

    shared = bootstrap.shared_units(mine, baseline)

    assert shared == UNITS[:2]
    assert bootstrap.drawn(bootstrap.units_of(mine)) != bootstrap.drawn(shared)


def test_the_draws_depend_only_on_the_units():
    # What makes a preset and the baseline comparable at all: the same unit
    # set gives the same resamples, whoever asks and in whatever order.
    first = CellBootstrap(_reporting(), {})
    second = CellBootstrap(_reporting(), {})

    assert first.drawn(UNITS) == second.drawn(UNITS)


def test_the_summary_does_not_depend_on_the_worker_count():
    metric = _Counted()
    cells = [
        _cell("base", UNITS),
        _cell("no_code", UNITS),
        _cell("no_perf", UNITS),
    ]
    rows = _rows(metric, cells)

    def summarised(workers: int) -> pd.DataFrame:
        return summarise(
            rows=rows,
            metrics_by_id={metric.id: metric},
            cell_contexts=cells,
            reporting=_reporting(workers=workers),
            run_id="r",
        )

    pd.testing.assert_frame_equal(summarised(1), summarised(3))


def test_every_reported_value_still_gets_its_own_row_and_interval():
    metric = _Counted()
    cells = [_cell("base", UNITS), _cell("no_code", UNITS)]

    summary = summarise(
        rows=_rows(metric, cells),
        metrics_by_id={metric.id: metric},
        cell_contexts=cells,
        reporting=_reporting(workers=1),
    )

    assert len(summary) == 6, "three values x two presets"
    assert set(summary["reported_value"]) == {"first", "second", "third"}
    assert set(summary["value_kind"]) == {KIND_RESULT, "input"}
    # The values differ from each other, so a shared call is not a shared
    # answer: caching the call must not collapse them onto one number.
    ablated = summary[summary["ablation"] == "no_code"]
    assert ablated["estimate"].nunique() > 1
    # And the paired difference reached every one of them.
    assert ablated["paired_n"].tolist() == [len(UNITS)] * 3
