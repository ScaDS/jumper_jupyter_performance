"""Speedup over the human fast solution ECCO pairs with the slow one.

``T(human fast) / T(suggestion)``: above 1 the rewrite beat the optimisation
a person submitted, below 1 it fell short of it. ECCO itself only reports the
speedup over the slow input; this is the comparison its pairs make possible.
Both sides are timed by cli.verify in one fresh process each, over the same
scaled inputs.
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


class ReferenceSpeedup(DeterministicMetric):
    spec = MetricSpec(
        id="reference_speedup",
        display_name="Speedup over Human Fast Solution",
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=(
            "geometric_mean_reference_speedup",
            "correct_suggestions",
        ),
        description=(
            "T(human fast) / T(suggestion), geometric mean over correct "
            "suggestions; above 1 beats the ECCO fast solution."
        ),
        supporting_values=("correct_suggestions",),
        ratio_values=("geometric_mean_reference_speedup",),
    )

    def compute(self, context, parameters: dict) -> dict:
        references = references_of(context, "ecco")
        if references is None:
            return {
                "geometric_mean_reference_speedup": None,
                "correct_suggestions": 0,
            }
        human = references.entries.get(HUMAN_FAST)
        baseline = human.time_s if human and human.ok else None
        correct = correct_suggestions(context)
        return {
            "geometric_mean_reference_speedup": geometric_mean(
                ratio(baseline, measurement.time_s) for measurement in correct
            ),
            "correct_suggestions": len(correct),
        }
