"""Speedup among the suggestions that were actually correct, with coverage.

The headline number of the whole experiment, and the one most easily abused:
condition hard enough and any preset looks excellent over the two suggestions
that survived. So coverage is reported as a first-class value beside it, never
as a footnote. A preset at 3.0x over 10% coverage and one at 1.8x over 90% are
not close, and the table has to show why.
"""

from __future__ import annotations

from jumper_ablations.metrics.base import (
    CATEGORY_SUGGESTIONS,
    METHOD_DETERMINISTIC,
    SCOPE_CELL,
    DeterministicMetric,
    MetricSpec,
)
from jumper_ablations.metrics.correctness import (
    correct_but_untimed,
    correct_speedups_in,
)
from jumper_ablations.metrics.verdicts import (
    ordered_verdicts,
)
from jumper_ablations.statistics import geometric_mean, median, rate


class CorrectnessConditionedSpeedup(DeterministicMetric):
    spec = MetricSpec(
        id="correctness_conditioned_speedup",
        display_name="Correctness-Conditioned Speedup / Coverage",
        value_names={
            "geometric_mean_speedup": "Geo-mean speedup, correct",
            "median_speedup": "Median speedup, correct",
            "coverage": "Correct share",
        },
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=(
            "geometric_mean_speedup",
            "median_speedup",
            "coverage",
            "correct_suggestions",
            "correct_but_untimed",
            "suggestions",
        ),
        description=(
            "Geometric mean and median speedup over correct suggestions only, "
            "with the share of all suggestions that qualified. The speedup is "
            "meaningless without the coverage and is never quoted alone."
        ),
        supporting_values=(
            "correct_suggestions",
            "correct_but_untimed",
            "suggestions",
        ),
        ratio_values=(
            "geometric_mean_speedup",
            "median_speedup",
        ),
    )

    def compute(self, context, parameters: dict) -> dict:
        correct = []
        total = 0
        untimed = 0
        for record in context.primary_benchmarks():
            correct.extend(correct_speedups_in(context, record))
            total += len(ordered_verdicts(record))
            untimed += correct_but_untimed(context, record)

        return {
            "geometric_mean_speedup": geometric_mean(correct),
            "median_speedup": median(correct),
            "coverage": rate(len(correct), total),
            "correct_suggestions": len(correct),
            "correct_but_untimed": untimed,
            "suggestions": total,
        }
