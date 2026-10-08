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
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from jumper_ablations.config.schema import ReportingConfig
from jumper_ablations.metrics.base import SCOPE_CELL, SCOPE_RUN, MetricRow
from jumper_ablations.metrics.context import CellContext, unit_key_of
from jumper_ablations.statistics import bootstrap_interval, clean, mean

logger = logging.getLogger("jumper_ablations")

SUMMARY_COLUMNS = (
    # First, because it is what makes a row self-describing once rows from
    # several runs sit in one table. Joining on the file's location instead
    # works exactly until the file is copied.
    "run_id",
    "usecase",
    "ablation",
    "category",
    "evaluation_method",
    "metric",
    "reported_value",
    # The spreadsheet's name for this value, empty where it has none. Kept
    # here and not in the per-unit file: it is a constant of the metric, and
    # repeating it on every one of a hundred thousand rows says nothing more.
    "metric_name",
    # Whether this row is the metric's answer or a quantity it was computed
    # from, so a reader can ask for results alone without knowing the names.
    "value_kind",
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
    run_id: str = "",
) -> pd.DataFrame:
    """One row per (usecase, ablation, metric, reported value).

    Usecases are summarised one at a time and may be summarised in parallel:
    nothing a cell-scope metric does crosses a usecase, and the resample
    cache that makes the cell path affordable is only useful within one.
    """
    parameters = parameters or {}
    frame = rows_frame(rows)
    if frame.empty:
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))

    usecases = sorted(frame["usecase"].unique())
    work = [
        (
            frame[frame["usecase"] == usecase],
            [
                context
                for context in cell_contexts
                if context.usecase_id == usecase
            ],
        )
        for usecase in usecases
    ]
    summary = []
    for produced in _map_usecases(
        work, metrics_by_id, reporting, parameters, run_id
    ):
        summary.extend(produced)
    return pd.DataFrame(summary, columns=list(SUMMARY_COLUMNS))


def _workers(reporting: ReportingConfig, usecases: int) -> int:
    """How many processes to summarise with. 0 in the config means choose.

    Capped low on purpose: the report is usually run interactively on a
    login node that other people are also using, and the first optimisation
    already removed an order of magnitude from the work.
    """
    if reporting.workers:
        return max(1, min(reporting.workers, usecases))
    return max(1, min(usecases, os.cpu_count() or 1, 4))


def _map_usecases(
    work: list, metrics_by_id: dict, reporting, parameters, run_id: str
) -> list[list[dict]]:
    """Summarise each usecase, in this process or in several.

    Returns a complete list rather than yielding as results arrive: a pool
    that fails halfway has to be retried from the beginning, and a caller
    that had already consumed part of the first attempt would then count
    those usecases twice.
    """
    workers = _workers(reporting, len(work))
    if workers > 1:
        logger.info(
            f"summarising {len(work)} usecase(s) on {workers} process(es)"
        )
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(
                        _summarise_usecase,
                        group,
                        contexts,
                        metrics_by_id,
                        reporting,
                        parameters,
                        run_id,
                    )
                    for group, contexts in work
                ]
                return [future.result() for future in futures]
        except Exception as failure:
            # A pool that cannot start - no fork, a sandbox, an unpicklable
            # metric someone has just added - is a reason to be slow, not a
            # reason to produce no report.
            logger.warning(f"falling back to one process: {failure}")

    return [
        _summarise_usecase(
            group, contexts, metrics_by_id, reporting, parameters, run_id
        )
        for group, contexts in work
    ]


def _summarise_usecase(
    frame: pd.DataFrame,
    cell_contexts: list,
    metrics_by_id: dict,
    reporting: ReportingConfig,
    parameters: dict,
    run_id: str,
) -> list[dict]:
    """Every summary row of one usecase."""
    contexts_by_cell = {
        (context.usecase_id, context.ablation_id): context
        for context in cell_contexts
    }
    bootstrap = CellBootstrap(reporting, parameters)

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
            bootstrap=bootstrap,
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
                "run_id": run_id,
                "usecase": usecase,
                "ablation": ablation,
                "category": metric.spec.category,
                "evaluation_method": metric.spec.evaluation_method,
                "metric": metric_id,
                "reported_value": reported_value,
                "metric_name": metric.spec.name_of(reported_value),
                "value_kind": metric.spec.kind_of(reported_value),
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
                    bootstrap=bootstrap,
                ),
                "gaps": values.get("gaps"),
            }
        )

    return summary


def _estimate(
    group: pd.DataFrame,
    metric,
    reported_value: str,
    context: CellContext | None,
    reporting: ReportingConfig,
    bootstrap: CellBootstrap,
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
        bootstrap=bootstrap,
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
    """Values keyed by the experimental unit, for pairing two presets."""
    paired = {}
    for unit_id, value in zip(group["unit_id"], group["value"]):
        key = unit_key_of(str(unit_id))
        if key is not None and pd.notna(value):
            paired[key] = float(value)
    return paired


def _draw_units(generator, units: tuple) -> list:
    """One bootstrap resample of the units, with replacement."""
    positions = generator.integers(0, len(units), size=len(units))
    return [units[position] for position in positions]


class CellBootstrap:
    """The resamples of one grid row, computed once and read many times.

    Every cell-scope number in the report comes from the same operation:
    rebuild a cell from a drawn set of units and recompute a metric on it.
    Done naively that operation runs once per *reported value*, and once
    again for the paired difference, so a metric that returns four numbers is
    computed four times on identical input and then twice more against the
    baseline. On the pilot that was 212,000 computations per usecase where
    19,200 carry all the information.

    Two things make the saving exact rather than approximate.

    A metric returns **all** of its reported values from one call, so caching
    the call and reading four keys out of it is the same arithmetic as four
    calls.

    And the draw depends only on the unit set: the generator is seeded per
    unit tuple, so a preset and the baseline that ran the same units get the
    same draws, and the interval and the paired difference share them. When
    the unit sets differ the tuples differ, each gets its own draws, and
    nothing is reused across populations that are not the same.
    """

    def __init__(self, reporting: ReportingConfig, parameters: dict):
        self._reporting = reporting
        self._parameters = parameters
        # Recomputing a metric is far more expensive than resampling a
        # number, so the cell-scope path uses a smaller draw than the
        # run-scope one and says so in the report rather than pretending to
        # ten thousand.
        self._draws = min(reporting.bootstrap.samples, 400)
        self._drawn: dict[tuple, list[list]] = {}
        self._values: dict[tuple, list[dict | None]] = {}
        self._points: dict[tuple, dict | None] = {}

    def units_of(self, context: CellContext | None) -> tuple:
        return () if context is None else context.unit_keys()

    def shared_units(
        self, context: CellContext | None, other: CellContext | None
    ) -> tuple:
        """The units both cells ran, in the first one's order."""
        if context is None or other is None:
            return ()
        theirs = set(self.units_of(other))
        return tuple(
            unit for unit in self.units_of(context) if unit in theirs
        )

    def drawn(self, units: tuple) -> list[list]:
        """The resamples of *units*, the same list every time it is asked."""
        if units not in self._drawn:
            generator = np.random.default_rng(
                self._reporting.bootstrap.seed
            )
            self._drawn[units] = [
                _draw_units(generator, units) for _ in range(self._draws)
            ]
        return self._drawn[units]

    def resampled_values(
        self, context: CellContext, metric, units: tuple
    ) -> list[dict | None]:
        """*metric* on each resample of *units*, one mapping per draw.

        None where the metric could not say anything about that resample,
        which is ordinary: a rank correlation needs two comparable points.
        """
        key = (context.usecase_id, context.ablation_id, metric.id, units)
        if key not in self._values:
            parameters = self._parameters.get(metric.id, {})
            computed: list[dict | None] = []
            for chosen in self.drawn(units):
                try:
                    computed.append(
                        metric.compute(
                            context.restricted_to(chosen), parameters
                        )
                    )
                except Exception:
                    computed.append(None)
            self._values[key] = computed
        return self._values[key]

    def on_units(
        self, context: CellContext, metric, units: tuple
    ) -> dict | None:
        """*metric* on one named set of units, outside the resamples.

        Cached like the resamples and for the same reason: the point estimate
        of the paired difference is asked for once per reported value, and a
        metric answers for all of them at once.
        """
        key = (context.usecase_id, context.ablation_id, metric.id, units)
        if key not in self._points:
            try:
                self._points[key] = metric.compute(
                    context.restricted_to(list(units)),
                    self._parameters.get(metric.id, {}),
                )
            except Exception:
                self._points[key] = None
        return self._points[key]


def _cell_paired_difference(
    metric,
    reported_value: str,
    context: CellContext | None,
    baseline_context: CellContext | None,
    reporting: ReportingConfig,
    bootstrap: CellBootstrap,
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
    shared = bootstrap.shared_units(context, baseline_context)
    if not shared:
        return dict(_NO_PAIRED_DIFFERENCE)

    point = _subtract(
        bootstrap.on_units(context, metric, shared),
        bootstrap.on_units(baseline_context, metric, shared),
        reported_value,
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

    mine = bootstrap.resampled_values(context, metric, shared)
    theirs = bootstrap.resampled_values(baseline_context, metric, shared)
    differences = [
        difference
        for value, baseline in zip(mine, theirs)
        if (difference := _subtract(value, baseline, reported_value))
        is not None
    ]

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


def _subtract(
    values: dict | None, baseline: dict | None, reported_value: str
) -> float | None:
    """One reported value of two already computed cells, subtracted."""
    if values is None or baseline is None:
        return None
    mine = values.get(reported_value)
    theirs = baseline.get(reported_value)
    if mine is None or theirs is None:
        return None
    return float(mine) - float(theirs)


def _cell_interval(
    metric,
    reported_value: str,
    context: CellContext | None,
    reporting: ReportingConfig,
    bootstrap: CellBootstrap,
) -> tuple[float | None, float | None]:
    """Bootstrap a cell-scope metric by resampling its generations.

    Recomputing the metric on each resample is the expensive but honest
    option: a pass@k interval taken over one number would be no interval at
    all. The resamples themselves come from the shared bootstrap, so the
    cost is paid once per metric rather than once per reported value.
    """
    if context is None:
        return (None, None)

    units = bootstrap.units_of(context)
    if len(units) < 2:
        return (None, None)

    numbers = clean(
        [
            None if values is None else values.get(reported_value)
            for values in bootstrap.resampled_values(context, metric, units)
        ]
    )
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
    bootstrap: CellBootstrap,
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
        bootstrap=bootstrap,
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
