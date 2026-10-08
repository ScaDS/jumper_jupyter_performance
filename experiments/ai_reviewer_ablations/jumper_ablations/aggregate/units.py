"""One value per experimental unit - the grain that can be pooled.

``metrics.csv`` carries two grains at once, and only one of them survives a
join across runs. A run-scope metric is already one value per reviewer
invocation. A cell-scope metric is not: by the time it reaches that file it
has reduced five repetitions to a single number, and the row says so in
``sample_size``. Averaging two runs' cell rows therefore answers nothing -
the weights are wrong - and the paired interval cannot be rebuilt from them
at all, because the paired bootstrap resamples units.

So this module writes the grain underneath: every metric evaluated on each
unit on its own, for both scopes, in one table keyed by
``(run_id, usecase, ablation, metric, reported_value, repetition,
generation)``. Two runs that asked the same question can be pooled by
concatenating it and recomputing, which is the only honest way to combine
them.

A per-unit value of a cell-scope metric is an *input*, never a result to
report. ``pass_at_k`` over a single generation is degenerate by construction,
and so is any rate whose denominator is one suggestion. The number belongs in
a resample, not in a table someone reads.
"""

from __future__ import annotations

import logging

import pandas as pd

from jumper_ablations.metrics.base import SCOPE_CELL, SCOPE_RUN, MetricRow
from jumper_ablations.metrics.context import CellContext, unit_key_of

logger = logging.getLogger("jumper_ablations")

UNIT_COLUMNS = (
    "run_id",
    "usecase",
    "ablation",
    "metric",
    "reported_value",
    "value_kind",
    "repetition",
    "generation",
    "scope",
    "category",
    "evaluation_method",
    "value",
    "note",
)


def unit_values(
    rows: list[MetricRow],
    metrics_by_id: dict,
    cell_contexts: list[CellContext],
    parameters: dict | None = None,
    run_id: str = "",
) -> pd.DataFrame:
    """Every metric value at unit grain, for both scopes."""
    parameters = parameters or {}
    collected = _from_run_scope(rows, metrics_by_id, run_id)
    collected.extend(
        _from_cell_scope(metrics_by_id, cell_contexts, parameters, run_id)
    )
    return pd.DataFrame(collected, columns=list(UNIT_COLUMNS))


def _from_run_scope(
    rows: list[MetricRow], metrics_by_id: dict, run_id: str
) -> list[dict]:
    """Run-scope rows carried over, since they are already per unit."""
    collected = []
    for row in rows:
        if row.scope != SCOPE_RUN:
            continue
        key = unit_key_of(row.unit_id)
        if key is None:
            continue
        collected.append(
            _entry(
                run_id=run_id,
                usecase=row.usecase,
                ablation=row.ablation,
                metric=row.metric,
                reported_value=row.reported_value,
                value_kind=row.value_kind,
                key=key,
                scope=row.scope,
                category=row.category,
                evaluation_method=row.evaluation_method,
                value=row.value,
                note=row.note,
            )
        )
    return collected


def _from_cell_scope(
    metrics_by_id: dict,
    cell_contexts: list[CellContext],
    parameters: dict,
    run_id: str,
) -> list[dict]:
    """Cell-scope metrics recomputed on one unit at a time.

    The same call the paired bootstrap makes, just with a single unit in the
    resample instead of a drawn list.
    """
    collected = []
    cell_metrics = [
        metric
        for metric in metrics_by_id.values()
        if metric.spec.scope == SCOPE_CELL
    ]
    for context in cell_contexts:
        for key in context.unit_keys():
            unit = context.restricted_to([key])
            for metric in cell_metrics:
                collected.extend(
                    _one_unit(metric, unit, key, parameters, run_id)
                )
    return collected


def _one_unit(
    metric, unit: CellContext, key: tuple[int, int], parameters, run_id: str
) -> list[dict]:
    """One metric on one unit, or the same rows with no value."""
    try:
        values = metric.compute(unit, parameters.get(metric.id, {}))
        note = ""
    except Exception as failure:
        # A metric that cannot say anything about a single unit is normal -
        # a correlation needs two points - and is not worth a traceback.
        logger.debug(f"{metric.id} on unit {key} of {unit.unit_id}: {failure}")
        values, note = {}, f"error: {failure}"
    return [
        _entry(
            run_id=run_id,
            usecase=unit.usecase_id,
            ablation=unit.ablation_id,
            metric=metric.id,
            reported_value=name,
            value_kind=metric.spec.kind_of(name),
            key=key,
            scope=metric.spec.scope,
            category=metric.spec.category,
            evaluation_method=metric.spec.evaluation_method,
            value=values.get(name),
            note=note,
        )
        for name in metric.spec.reported_values
    ]


def _entry(
    run_id: str,
    usecase: str,
    ablation: str,
    metric: str,
    reported_value: str,
    value_kind: str,
    key: tuple[int, int],
    scope: str,
    category: str,
    evaluation_method: str,
    value,
    note: str,
) -> dict:
    repetition, generation = key
    return {
        "run_id": run_id,
        "usecase": usecase,
        "ablation": ablation,
        "metric": metric,
        "reported_value": reported_value,
        "value_kind": value_kind,
        "repetition": repetition,
        "generation": generation,
        "scope": scope,
        "category": category,
        "evaluation_method": evaluation_method,
        "value": value,
        "note": note,
    }
