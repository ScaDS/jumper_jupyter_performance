"""Point estimates, intervals, and the difference from the baseline preset.

Two shapes of metric arrive here and they need different treatment.

A **run-scope** metric produced one value per reviewer invocation, so its
distribution is right there: the estimate is the mean of those values and the
interval is bootstrapped over them.

A **cell-scope** metric already reduced a whole grid cell to one number -
pass@k over ten generations is not ten numbers - so there is nothing to
resample at this level. The interval is obtained by resampling the
*generations* and recomputing the metric on each resample, which is the only
way to get an interval that means what it says.

The comparison against the baseline is paired wherever pairing exists. Every
preset was asked for generation 1, generation 2 and so on under the same
conditions, so the difference of the paired values has far less variance than
the difference of the two means.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from jumper_ablations.config.schema import ReportingConfig
from jumper_ablations.metrics.base import SCOPE_CELL, SCOPE_RUN, MetricRow
from jumper_ablations.metrics.context import CellContext
from jumper_ablations.statistics import bootstrap_interval, clean, mean

logger = logging.getLogger("jumper_ablations")

SUMMARY_COLUMNS = (
    "usecase",
    "ablation",
    "category",
    "evaluation_method",
    "metric",
    "reported_value",
    "estimate",
    "ci_low",
    "ci_high",
    "n",
    "baseline_estimate",
    "delta_vs_baseline",
    "paired_delta",
    "gaps",
)


def rows_frame(rows: list[MetricRow]) -> pd.DataFrame:
    """Every metric value as a flat table - the raw material of the report."""
    return pd.DataFrame([row.__dict__ for row in rows])


def summarise(
    rows: list[MetricRow],
    metrics_by_id: dict,
    cell_contexts: list[CellContext],
    reporting: ReportingConfig,
    parameters: dict | None = None,
) -> pd.DataFrame:
    """One row per (usecase, ablation, metric, reported value)."""
    parameters = parameters or {}
    frame = rows_frame(rows)
    if frame.empty:
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))

    contexts_by_cell = {
        (context.usecase_id, context.ablation_id): context
        for context in cell_contexts
    }

    estimates: dict[tuple, dict] = {}
    for key, group in frame.groupby(
        ["usecase", "ablation", "metric", "reported_value"],
        sort=True,
    ):
        usecase, ablation, metric_id, reported_value = key
        metric = metrics_by_id.get(metric_id)
        if metric is None:
            continue
        estimates[key] = _estimate(
            group=group,
            metric=metric,
            reported_value=reported_value,
            context=contexts_by_cell.get((usecase, ablation)),
            reporting=reporting,
            metric_parameters=parameters.get(metric_id, {}),
        )

    summary = []
    for key, values in estimates.items():
        usecase, ablation, metric_id, reported_value = key
        metric = metrics_by_id[metric_id]
        baseline_key = (usecase, reporting.baseline_ablation, metric_id, reported_value)
        baseline = estimates.get(baseline_key, {})
        summary.append(
            {
                "usecase": usecase,
                "ablation": ablation,
                "category": metric.spec.category,
                "evaluation_method": metric.spec.evaluation_method,
                "metric": metric_id,
                "reported_value": reported_value,
                "estimate": values.get("estimate"),
                "ci_low": values.get("ci_low"),
                "ci_high": values.get("ci_high"),
                "n": values.get("n"),
                "baseline_estimate": baseline.get("estimate"),
                "delta_vs_baseline": _difference(
                    values.get("estimate"),
                    baseline.get("estimate"),
                ),
                "paired_delta": _paired_delta(
                    values.get("per_generation"),
                    baseline.get("per_generation"),
                ),
                "gaps": values.get("gaps"),
            }
        )

    return pd.DataFrame(summary, columns=list(SUMMARY_COLUMNS))


def _estimate(
    group: pd.DataFrame,
    metric,
    reported_value: str,
    context: CellContext | None,
    reporting: ReportingConfig,
    metric_parameters: dict,
) -> dict:
    values = clean(group["value"].tolist())
    gaps = int(group["value"].isna().sum())

    if metric.spec.scope == SCOPE_RUN:
        low, high = bootstrap_interval(
            values,
            samples=reporting.bootstrap.samples,
            confidence=reporting.bootstrap.confidence,
            seed=reporting.bootstrap.seed,
        )
        return {
            "estimate": mean(values),
            "ci_low": low,
            "ci_high": high,
            "n": len(values),
            "gaps": gaps,
            "per_generation": _per_generation(group),
        }

    if metric.spec.scope != SCOPE_CELL:
        raise ValueError(f"{metric.id}: unhandled scope {metric.spec.scope!r}")

    estimate = values[0] if values else None
    low, high = _cell_interval(
        metric=metric,
        reported_value=reported_value,
        context=context,
        reporting=reporting,
        metric_parameters=metric_parameters,
    )
    return {
        "estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "n": context.sample_size if context else 0,
        "gaps": gaps,
        "per_generation": None,
    }


def _per_generation(group: pd.DataFrame) -> dict:
    """Values keyed by generation, so two presets can be compared pairwise.

    The unit id ends in the phase and mode; the generation is the ``gNN``
    field, which is what pairs a run of one preset with a run of another.
    """
    paired = {}
    for unit_id, value in zip(group["unit_id"], group["value"]):
        generation = _generation_of(str(unit_id))
        if generation is not None and pd.notna(value):
            paired[generation] = float(value)
    return paired


def _generation_of(unit_id: str) -> int | None:
    for part in unit_id.split("__"):
        if part.startswith("g") and part[1:].isdigit():
            return int(part[1:])
    return None


def _cell_interval(
    metric,
    reported_value: str,
    context: CellContext | None,
    reporting: ReportingConfig,
    metric_parameters: dict,
) -> tuple[float | None, float | None]:
    """Bootstrap a cell-scope metric by resampling its generations.

    Recomputing the metric on each resample is the expensive but honest
    option: a pass@k interval taken over one number would be no interval at
    all.
    """
    if context is None:
        return (None, None)

    generations = sorted(
        {
            record.identity.generation
            for record in context.records
            if record.identity.generation
        }
    )
    if len(generations) < 2:
        return (None, None)

    generator = np.random.default_rng(reporting.bootstrap.seed)
    # Recomputing a metric is far more expensive than resampling a number, so
    # this uses a smaller draw than the run-scope path and says so in the
    # report rather than pretending to ten thousand.
    draws = min(reporting.bootstrap.samples, 400)
    estimates = []
    for _ in range(draws):
        chosen = generator.choice(generations, size=len(generations), replace=True)
        resampled = CellContext(
            usecase_id=context.usecase_id,
            ablation_id=context.ablation_id,
            records=tuple(
                record
                for generation in chosen
                for record in context.generation(int(generation))
            ),
            usecase=context.usecase,
        )
        try:
            values = metric.compute(resampled, metric_parameters)
        except Exception:
            continue
        estimates.append(values.get(reported_value))

    numbers = clean(estimates)
    if len(numbers) < 2:
        return (None, None)
    tail = (1.0 - reporting.bootstrap.confidence) / 2.0
    return (
        float(np.percentile(numbers, 100.0 * tail)),
        float(np.percentile(numbers, 100.0 * (1.0 - tail))),
    )


def _difference(value, baseline) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


def _paired_delta(values: dict | None, baseline: dict | None) -> float | None:
    """Mean difference over the generations both presets have.

    None when there is nothing to pair - which is the honest answer, not zero.
    """
    if not values or not baseline:
        return None
    shared = sorted(set(values) & set(baseline))
    if not shared:
        return None
    return mean([values[generation] - baseline[generation] for generation in shared])
