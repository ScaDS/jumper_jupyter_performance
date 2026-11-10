"""Memory reduction over the human fast solution ECCO pairs with the slow one.

``M(human fast) / M(suggestion)``, peak resident memory of one fresh process
each: above 1 the rewrite needs less memory than the optimisation a person
submitted.

Split as `reference_speedup` is:

``geometric_mean_reference_memory_reduction`` - **Memory reduction vs human** -
over every correct suggestion of the cell.

``best_reference_memory_reduction`` - **Best memory reduction vs human** - the
leanest correct suggestion of the whole cell, across every generation and
repetition. An extremum, so it rises with the number of draws and cannot be
compared between presets drawn a different number of times.
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
        value_names={
            "geometric_mean_reference_memory_reduction": "Memory reduction vs human",
            "best_reference_memory_reduction": "Best memory reduction vs human",
        },
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=(
            "geometric_mean_reference_memory_reduction",
            "best_reference_memory_reduction",
            "correct_suggestions",
        ),
        description=(
            "M(human fast) / M(suggestion); above 1 needs less memory than "
            "the ECCO fast solution. Geometric mean over correct suggestions, "
            "and the leanest correct suggestion of the whole cell."
        ),
        supporting_values=("correct_suggestions",),
        ratio_values=(
            "geometric_mean_reference_memory_reduction",
            "best_reference_memory_reduction",
        ),
    )

    def compute(self, context, parameters: dict) -> dict:
        references = references_of(context, "ecco")
        if references is None:
            return {
                "geometric_mean_reference_memory_reduction": None,
                "best_reference_memory_reduction": None,
                "correct_suggestions": 0,
            }
        human = references.entries.get(HUMAN_FAST)
        baseline = human.peak_memory_bytes if human and human.ok else None
        correct = correct_suggestions(context)
        reductions = [
            value
            for value in (
                ratio(baseline, measurement.peak_memory_bytes)
                for measurement in correct
            )
            if value is not None
        ]
        return {
            "geometric_mean_reference_memory_reduction": geometric_mean(reductions),
            "best_reference_memory_reduction": (
                max(reductions) if reductions else None
            ),
            "correct_suggestions": len(correct),
        }
