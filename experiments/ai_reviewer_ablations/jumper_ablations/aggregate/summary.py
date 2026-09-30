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
    "paired_delta_ci_low",
    "paired_delta_ci_high",
    "paired_n",
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
        baseline_key = (
            usecase,
            reporting.baseline_ablation,
            metric_id,
            reported_value,
        )
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
                **_difference_from_baseline(
                    metric=metric,
                    reported_value=reported_value,
                    values=values,
                    baseline=baseline,
                    context=contexts_by_cell.get((usecase, ablation)),
                    baseline_context=contexts_by_cell.get(
                        (usecase, reporting.baseline_ablation)
                    ),
                    reporting=reporting,
                    metric_parameters=parameters.get(metric_id, {}),
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
            "per_unit": _per_unit(group),
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
        "per_unit": None,
    }


def _per_unit(group: pd.DataFrame) -> dict:
    """Values keyed by the experimental unit, for pairing two presets.

    The unit is the *pass* and the generation within it - ``r00`` and ``g01``
    in the unit id - not the generation alone. Keying on the generation made
    a second repetition silently overwrite the first, so with
    ``repetitions > 1`` half the data left the comparison without a trace.
    """
    paired = {}
    for unit_id, value in zip(group["unit_id"], group["value"]):
        key = _unit_key_of(str(unit_id))
        if key is not None and pd.notna(value):
            paired[key] = float(value)
    return paired


def _unit_key_of(unit_id: str) -> tuple[int, int] | None:
    """``(repetition, generation)`` out of a record id, or None."""
    repetition, generation = None, None
    for part in unit_id.split("__"):
        if part.startswith("r") and part[1:].isdigit():
            repetition = int(part[1:])
        elif part.startswith("g") and part[1:].isdigit():
            generation = int(part[1:])
    if generation is None:
        return None
    return (repetition if repetition is not None else 0, generation)


def _draw_units(generator, units: tuple) -> list:
    """One bootstrap resample of the units, with replacement."""
    positions = generator.integers(0, len(units), size=len(units))
    return [units[position] for position in positions]


def _resampled(context: CellContext, chosen: list) -> CellContext:
    """The same cell, rebuilt from a resampled list of units."""
    return CellContext(
        usecase_id=context.usecase_id,
        ablation_id=context.ablation_id,
        records=tuple(
            record for unit in chosen for record in context.unit(unit)
        ),
        usecase=context.usecase,
    )


def _cell_paired_difference(
    metric,
    reported_value: str,
    context: CellContext | None,
    baseline_context: CellContext | None,
    reporting: ReportingConfig,
    metric_parameters: dict,
) -> dict:
    """The difference from the baseline for a metric that reduces a whole
    cell to one number, and an interval for that difference.

    A cell-scope metric has no per-unit value to subtract - pass@k over ten
    generations is one number, not ten - so the difference is bootstrapped
    instead: draw a set of units, rebuild *both* presets from the same drawn
    units, recompute both metrics, and take the difference. Drawing the same
    units for both is what makes it paired, and pairing is what removes the
    payload and the machine from the comparison.

    Without this the report had no interval at all for any deterministic
    metric, and drew the preset's own spread around a bare difference of two
    point estimates.
    """
    if context is None or baseline_context is None:
        return dict(_NO_PAIRED_DIFFERENCE)

    shared = tuple(
        unit
        for unit in context.unit_keys()
        if unit in set(baseline_context.unit_keys())
    )
    if not shared:
        return dict(_NO_PAIRED_DIFFERENCE)

    point = _difference_on(
        metric,
        reported_value,
        context,
        baseline_context,
        list(shared),
        metric_parameters,
    )
    if point is None:
        return dict(_NO_PAIRED_DIFFERENCE)

    result = {
        "paired_delta": point,
        "paired_delta_ci_low": None,
        "paired_delta_ci_high": None,
        "paired_n": len(shared),
    }
    if len(shared) < 2:
        return result

    generator = np.random.default_rng(reporting.bootstrap.seed)
    draws = min(reporting.bootstrap.samples, 400)
    differences = []
    for _ in range(draws):
        chosen = _draw_units(generator, shared)
        difference = _difference_on(
            metric,
            reported_value,
            context,
            baseline_context,
            chosen,
            metric_parameters,
        )
        if difference is not None:
            differences.append(difference)

    if len(differences) < 2:
        return result
    tail = (1.0 - reporting.bootstrap.confidence) / 2.0
    result["paired_delta_ci_low"] = float(
        np.percentile(differences, 100.0 * tail)
    )
    result["paired_delta_ci_high"] = float(
        np.percentile(differences, 100.0 * (1.0 - tail))
    )
    return result


def _difference_on(
    metric,
    reported_value: str,
    context: CellContext,
    baseline_context: CellContext,
    chosen: list,
    metric_parameters: dict,
) -> float | None:
    """Both presets recomputed on the same units, then subtracted."""
    try:
        value = metric.compute(
            _resampled(context, chosen), metric_parameters
        ).get(reported_value)
        baseline = metric.compute(
            _resampled(baseline_context, chosen), metric_parameters
        ).get(reported_value)
    except Exception:
        return None
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


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

    units = context.unit_keys()
    if len(units) < 2:
        return (None, None)

    generator = np.random.default_rng(reporting.bootstrap.seed)
    # Recomputing a metric is far more expensive than resampling a number, so
    # this uses a smaller draw than the run-scope path and says so in the
    # report rather than pretending to ten thousand.
    draws = min(reporting.bootstrap.samples, 400)
    estimates = []
    for _ in range(draws):
        chosen = _draw_units(generator, units)
        try:
            values = metric.compute(
                _resampled(context, chosen), metric_parameters
            )
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


def _difference_from_baseline(
    metric,
    reported_value: str,
    values: dict,
    baseline: dict,
    context,
    baseline_context,
    reporting: ReportingConfig,
    metric_parameters: dict,
) -> dict:
    """The paired difference, by whichever route this metric's shape allows.

    A run-scope metric has one value per unit, so the differences can be
    taken directly. A cell-scope metric has one value per cell, so both
    presets are recomputed on the same resampled units instead. Either way
    the report gets a difference and an interval that is about that
    difference - previously only the first kind had a paired delta at all,
    and every deterministic metric is of the second kind.
    """
    if metric.spec.scope == SCOPE_RUN:
        return _paired_difference(
            values.get("per_unit"), baseline.get("per_unit"), reporting
        )
    return _cell_paired_difference(
        metric=metric,
        reported_value=reported_value,
        context=context,
        baseline_context=baseline_context,
        reporting=reporting,
        metric_parameters=metric_parameters,
    )


_NO_PAIRED_DIFFERENCE = {
    "paired_delta": None,
    "paired_delta_ci_low": None,
    "paired_delta_ci_high": None,
    "paired_n": 0,
}


def _paired_difference(
    values: dict | None,
    baseline: dict | None,
    reporting: ReportingConfig,
) -> dict:
    """The mean paired difference from the baseline, and its own interval.

    The interval is resampled from the *differences*, which is the only thing
    it can be a statement about. The report previously drew this preset's own
    interval around the difference: that width describes how much this
    preset's values scatter, says nothing about the baseline's, and is
    therefore neither a paired nor an unpaired interval for the difference.
    Since the notebook tells the reader that an interval crossing zero means
    no effect, the wrong width decides conclusions in both directions.

    Pairing is what makes the difference worth measuring: both presets were
    asked for the same unit under the same seed, so whatever the payload and
    the machine contributed is in both values and cancels.
    """
    if not values or not baseline:
        return dict(_NO_PAIRED_DIFFERENCE)
    shared = sorted(set(values) & set(baseline))
    if not shared:
        return dict(_NO_PAIRED_DIFFERENCE)

    differences = [values[unit] - baseline[unit] for unit in shared]
    low, high = bootstrap_interval(
        differences,
        samples=reporting.bootstrap.samples,
        confidence=reporting.bootstrap.confidence,
        seed=reporting.bootstrap.seed,
    )
    return {
        "paired_delta": mean(differences),
        "paired_delta_ci_low": low,
        "paired_delta_ci_high": high,
        "paired_n": len(differences),
    }
