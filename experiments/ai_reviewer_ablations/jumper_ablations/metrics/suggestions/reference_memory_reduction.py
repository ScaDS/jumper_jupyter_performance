"""Memory reduction over the human fast solution ECCO pairs with the slow one.

``M(human fast) / M(suggestion)``, peak resident memory of one fresh process
each: above 1 the rewrite needs less memory than the optimisation a person
submitted.
"""

from __future__ import annotations

from jumper_ablations.metrics.base import (
    CATEGORY_SUGGESTIONS,
    METHOD_DETERMINISTIC,
    SCOPE_CELL,
    DeterministicMetric,
    MetricSpec,
)
from jumper_ablations.statistics import geometric_mean
from jumper_ablations.verification.schema import HUMAN_FAST
from jumper_ablations.verification.views import (
    correct_suggestions,
    ratio,
    references_of,
)


class ReferenceMemoryReduction(DeterministicMetric):
    spec = MetricSpec(
        id="reference_memory_reduction",
        display_name="Memory Reduction over Human Fast Solution",
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=(
            "geometric_mean_reference_memory_reduction",
            "correct_suggestions",
        ),
        description=(
            "M(human fast) / M(suggestion), geometric mean over correct "
            "suggestions; above 1 needs less memory than the ECCO fast solution."
        ),
        supporting_values=("correct_suggestions",),
        ratio_values=("geometric_mean_reference_memory_reduction",),
    )

    def compute(self, context, parameters: dict) -> dict:
        references = references_of(context, "ecco")
        if references is None:
            return {
                "geometric_mean_reference_memory_reduction": None,
                "correct_suggestions": 0,
            }
        human = references.entries.get(HUMAN_FAST)
        baseline = human.peak_memory_bytes if human and human.ok else None
        correct = correct_suggestions(context)
        return {
            "geometric_mean_reference_memory_reduction": geometric_mean(
                ratio(baseline, measurement.peak_memory_bytes)
                for measurement in correct
            ),
            "correct_suggestions": len(correct),
        }
