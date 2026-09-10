"""Best suggestion speedup: how good is the best option a preset produced?

The number a user actually experiences, because a user picks one suggestion
and it is rarely the mediocre one. Reported over correct suggestions as well
as over all measured ones, since a preset whose best option is fast and wrong
has not helped anybody.
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
from jumper_ablations.statistics import mean, median


class BestSpeedup(DeterministicMetric):
    spec = MetricSpec(
        id="best_speedup",
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=(
            "mean_best_speedup",
            "median_best_speedup",
            "mean_best_correct_speedup",
            "median_best_correct_speedup",
            "generations_with_a_measurement",
        ),
        description=(
            "Mean and median, over generations, of the fastest suggestion each "
            "generation produced - once over everything that ran, once over "
            "what also produced matching results."
        ),
    )

    def compute(self, context, parameters: dict) -> dict:
        best_measured = []
        best_correct = []
        for record in context.primary_benchmarks():
            measured = measured_speedups(record)
            if measured:
                best_measured.append(max(measured))
            correct = correct_speedups(record)
            if correct:
                best_correct.append(max(correct))

        return {
            "mean_best_speedup": mean(best_measured),
            "median_best_speedup": median(best_measured),
            "mean_best_correct_speedup": mean(best_correct),
            "median_best_correct_speedup": median(best_correct),
            "generations_with_a_measurement": len(best_measured),
        }
