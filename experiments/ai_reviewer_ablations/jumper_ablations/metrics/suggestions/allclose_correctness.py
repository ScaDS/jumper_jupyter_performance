"""KernelBench's correctness: outputs within tolerance on fresh inputs.

A suggestion is correct when its ``Model`` agrees with the original on every
one of ``correctness_trials`` random inputs under
``torch.allclose(atol, rtol)`` - elementwise ``|a - b| <= atol + rtol * |b|``.
Unlike the reviewer's fingerprint this compares every element, not five
statistics, and tolerates the rounding a fused or reordered kernel produces.
A suggestion that failed, timed out or no longer defines ``Model`` is
incorrect, as a compile failure is in KernelBench.
"""

from __future__ import annotations

from jumper_ablations.metrics.base import (
    CATEGORY_SUGGESTIONS,
    METHOD_DETERMINISTIC,
    SCOPE_CELL,
    DeterministicMetric,
    MetricSpec,
)
from jumper_ablations.statistics import rate
from jumper_ablations.verification.views import (
    references_of,
    verified_suggestions,
)


class AllcloseCorrectness(DeterministicMetric):
    spec = MetricSpec(
        id="allclose_correctness",
        display_name="Correctness (allclose)",
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=(
            "allclose_rate",
            "suggestions",
        ),
        description=(
            "Share of suggestions whose Model matches the original under "
            "torch.allclose on every random input (KernelBench)."
        ),
        supporting_values=("suggestions",),
    )

    def compute(self, context, parameters: dict) -> dict:
        if references_of(context, "kernelbench") is None:
            return {"allclose_rate": None, "suggestions": 0}
        measured = verified_suggestions(context)
        correct = sum(
            1 for measurement in measured if measurement.ok and measurement.correct
        )
        return {
            "allclose_rate": rate(correct, len(measured)),
            "suggestions": len(measured),
        }
