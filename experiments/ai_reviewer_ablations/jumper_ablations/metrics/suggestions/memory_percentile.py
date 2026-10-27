"""ECCO's Memory %: the share of submissions that need more memory.

``|{s : M(s) > M(suggestion)}| / |submissions|``, peak resident memory of
one fresh process each, over the same spectrum as Runtime %. A submission
that timed out never reported a peak, so it is left out here rather than
guessed.
"""

from __future__ import annotations

from jumper_ablations.metrics.base import (
    CATEGORY_SUGGESTIONS,
    METHOD_DETERMINISTIC,
    SCOPE_CELL,
    DeterministicMetric,
    MetricSpec,
)
from jumper_ablations.metrics.suggestions.runtime_percentile import percentile
from jumper_ablations.statistics import mean
from jumper_ablations.verification.views import (
    correct_suggestions,
    references_of,
)


class MemoryPercentile(DeterministicMetric):
    spec = MetricSpec(
        id="memory_percentile",
        display_name="Memory %",
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=(
            "mean_memory_percentile",
            "correct_suggestions",
            "submissions",
        ),
        description=(
            "Percentage of CodeNet submissions using more peak memory than "
            "the suggestion, mean over correct suggestions (ECCO)."
        ),
        supporting_values=("correct_suggestions", "submissions"),
    )

    def compute(self, context, parameters: dict) -> dict:
        references = references_of(context, "ecco")
        if references is None:
            return {
                "mean_memory_percentile": None,
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
        correct = correct_suggestions(context)
        return {
            "mean_memory_percentile": mean(
                percentile(values, measurement.peak_memory_bytes)
                for measurement in correct
            ),
            "correct_suggestions": len(correct),
            "submissions": len(values),
        }
