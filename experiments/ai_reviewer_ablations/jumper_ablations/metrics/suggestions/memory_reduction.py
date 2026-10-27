"""ECCO's Memory Reduction: how much less memory the rewrite needs.

ECCO defines it as ``Memory(p_in) / Memory(p_out)`` - the slow input program
over the generated one - measured on the private tests. Here both sides are
the peak resident set of one fresh process running the notebook prefix and
the program over the usecase's scaled inputs, as cli.verify measures them.
Only correct suggestions count: a program that prints something else has not
reduced anything.
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
from jumper_ablations.verification.schema import ORIGINAL
from jumper_ablations.verification.views import (
    correct_suggestions,
    ratio,
    references_of,
)


class MemoryReduction(DeterministicMetric):
    spec = MetricSpec(
        id="memory_reduction",
        display_name="Memory Reduction",
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=(
            "geometric_mean_memory_reduction",
            "correct_suggestions",
        ),
        description=(
            "M(original) / M(suggestion), peak resident memory of one fresh "
            "process each, geometric mean over correct suggestions (ECCO)."
        ),
        supporting_values=("correct_suggestions",),
        ratio_values=("geometric_mean_memory_reduction",),
    )

    def compute(self, context, parameters: dict) -> dict:
        references = references_of(context, "ecco")
        if references is None:
            return {
                "geometric_mean_memory_reduction": None,
                "correct_suggestions": 0,
            }
        original = references.entries.get(ORIGINAL)
        baseline = original.peak_memory_bytes if original else None
        correct = correct_suggestions(context)
        return {
            "geometric_mean_memory_reduction": geometric_mean(
                ratio(baseline, measurement.peak_memory_bytes)
                for measurement in correct
            ),
            "correct_suggestions": len(correct),
        }
