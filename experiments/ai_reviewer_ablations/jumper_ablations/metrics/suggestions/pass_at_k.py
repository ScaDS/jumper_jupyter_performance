"""pass@1 and pass@k over correct suggestions.

pass@1 asks whether the option the model put first is one a user could
actually adopt; pass@k asks whether any of them was. Both require the
suggestion to have run *and* to have produced matching results, so a
suggestion that is fast and wrong passes neither.

`k` is not a constant here. It is however many options the response actually
contained, which is why `requested_vs_returned` is reported next to this.
"""
from __future__ import annotations

from jumper_ablations.metrics.base import (
    CATEGORY_SUGGESTIONS,
    METHOD_DETERMINISTIC,
    SCOPE_CELL,
    DeterministicMetric,
    MetricSpec,
)
from jumper_ablations.metrics.verdicts import is_correct, ordered_verdicts
from jumper_ablations.statistics import mean, rate


class PassAtK(DeterministicMetric):
    spec = MetricSpec(
        id="pass_at_k",
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=("pass_at_1", "pass_at_k", "mean_k", "generations"),
        description=(
            "Fraction of generations whose first-ranked suggestion was "
            "correct, and whose response contained at least one correct "
            "suggestion. `mean_k` is the average number of options a response "
            "held."
        ),
    )

    def compute(self, context, parameters: dict) -> dict:
        first_correct = 0
        any_correct = 0
        sizes = []
        generations = 0

        for record in context.primary_benchmarks():
            verdicts = ordered_verdicts(record)
            if not verdicts:
                continue
            generations += 1
            sizes.append(len(verdicts))
            first_correct += is_correct(verdicts[0])
            any_correct += any(is_correct(verdict) for verdict in verdicts)

        return {
            "pass_at_1": rate(first_correct, generations),
            "pass_at_k": rate(any_correct, generations),
            "mean_k": mean(sizes),
            "generations": generations,
        }
