"""Mean and median speedup over every suggestion a preset produced.

The counterpart to best-of: it says what a whole set of options is worth
rather than what its winner is worth. A preset that produces one excellent
suggestion and four useless ones reads very differently here than it does in
`best_speedup`, and the difference is the point.
"""
from __future__ import annotations

from jumper_ablations.metrics.base import (
    CATEGORY_SUGGESTIONS,
    METHOD_DETERMINISTIC,
    SCOPE_CELL,
    DeterministicMetric,
    MetricSpec,
)
from jumper_ablations.metrics.verdicts import correct_speedups, measured_speedups
from jumper_ablations.statistics import geometric_mean, mean, median


class MeanMedianSpeedup(DeterministicMetric):
    spec = MetricSpec(
        id="mean_median_speedup",
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=(
            "mean_speedup",
            "median_speedup",
            "geometric_mean_speedup",
            "measured_suggestions",
        ),
        description=(
            "Arithmetic mean, median and geometric mean of the speedup of "
            "every suggestion that ran, pooled over the generations of this "
            "preset. The geometric mean is the one to quote: these are ratios."
        ),
    )

    def compute(self, context, parameters: dict) -> dict:
        pooled = []
        for record in context.primary_benchmarks():
            pooled.extend(measured_speedups(record))

        return {
            "mean_speedup": mean(pooled),
            "median_speedup": median(pooled),
            "geometric_mean_speedup": geometric_mean(pooled),
            "measured_suggestions": len(pooled),
        }


class MeanMedianCorrectSpeedup(DeterministicMetric):
    spec = MetricSpec(
        id="mean_median_correct_speedup",
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=(
            "mean_speedup",
            "median_speedup",
            "geometric_mean_speedup",
            "correct_suggestions",
        ),
        description=(
            "The same three averages over the suggestions that also produced "
            "matching results."
        ),
    )

    def compute(self, context, parameters: dict) -> dict:
        pooled = []
        for record in context.primary_benchmarks():
            pooled.extend(correct_speedups(record))

        return {
            "mean_speedup": mean(pooled),
            "median_speedup": median(pooled),
            "geometric_mean_speedup": geometric_mean(pooled),
            "correct_suggestions": len(pooled),
        }
