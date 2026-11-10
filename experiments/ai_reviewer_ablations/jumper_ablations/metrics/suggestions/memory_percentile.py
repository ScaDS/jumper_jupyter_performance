"""ECCO's Memory %: the share of submissions that need more memory.

``|{s : M(s) > M(suggestion)}| / |submissions|``, peak resident memory of
one fresh process each, over the same spectrum as Runtime %. A submission
that timed out never reported a peak, so it is left out here rather than
guessed.

Split exactly as Runtime % is, and for the same reason: a user applies one
suggestion out of one response, so each generation is reduced to its best
correct suggestion first.

``mean_memory_percentile`` - **Mean Memory %** - the mean of those: what a
generation of this preset achieves, typically.

``best_memory_percentile`` - **Best Memory %** - the largest of them, not
averaged: the best this preset was ever seen to do. An extremum over the whole
cell, so it rises with the number of generations and cannot be compared
between presets drawn a different number of times.
"""

from __future__ import annotations

from jumper_ablations.metrics.base import (
    CATEGORY_SUGGESTIONS,
    METHOD_DETERMINISTIC,
    SCOPE_CELL,
    DeterministicMetric,
    MetricSpec,
)
from jumper_ablations.metrics.suggestions.runtime_percentile import (
    best_per_review,
    percentile,
)
from jumper_ablations.statistics import mean
from jumper_ablations.verification.views import (
    correct_suggestions,
    references_of,
)


class MemoryPercentile(DeterministicMetric):
    spec = MetricSpec(
        id="memory_percentile",
        display_name="Memory %",
        value_names={
            "mean_memory_percentile": "Mean Memory %",
            "best_memory_percentile": "Best Memory %",
        },
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=(
            "mean_memory_percentile",
            "best_memory_percentile",
            "correct_suggestions",
            "submissions",
        ),
        description=(
            "Percentage of CodeNet submissions using more peak memory than "
            "the suggestion (ECCO), over each generation's best correct "
            "suggestion: their mean, and the single largest of them."
        ),
        supporting_values=("correct_suggestions", "submissions"),
    )

    def compute(self, context, parameters: dict) -> dict:
        references = references_of(context, "ecco")
        if references is None:
            return {
                "mean_memory_percentile": None,
                "best_memory_percentile": None,
                "correct_suggestions": 0,
                "submissions": 0,
            }
        values = [
            float(measurement.peak_memory_bytes)
            for measurement in references.population
            if measurement.ok
            and measurement.correct
            and measurement.peak_memory_bytes is not None
        ]
        best = best_per_review(context, values, "peak_memory_bytes")
        return {
            "mean_memory_percentile": mean(best),
            "best_memory_percentile": max(best) if best else None,
            "correct_suggestions": len(correct_suggestions(context)),
            "submissions": len(values),
        }
