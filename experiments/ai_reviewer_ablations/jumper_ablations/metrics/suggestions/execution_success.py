"""Did the suggestions run - before the repair loop, and after it?

Two numbers, and the gap between them is the interesting one. A preset whose
suggestions only run after three rounds of repair is being carried by the
repair loop, and reporting only the post-repair rate would hide that.

"Before repair" is read off the attempt counter: the reviewer records how many
code versions a suggestion needed, so a suggestion that succeeded on its first
version is one that needed no repair.
"""
from __future__ import annotations

from jumper_ablations.metrics.base import (
    CATEGORY_SUGGESTIONS,
    METHOD_DETERMINISTIC,
    SCOPE_CELL,
    DeterministicMetric,
    MetricSpec,
)
from jumper_ablations.metrics.verdicts import ordered_verdicts, ran
from jumper_ablations.statistics import rate


class ExecutionSuccess(DeterministicMetric):
    spec = MetricSpec(
        id="execution_success",
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=(
            "pre_repair_success_rate",
            "post_repair_success_rate",
            "repair_uplift",
            "suggestions",
        ),
        description=(
            "Fraction of suggestions that ran on their first version, the "
            "fraction that ran after repairs, and the difference."
        ),
    )

    def compute(self, context, parameters: dict) -> dict:
        total = 0
        first_try = 0
        eventually = 0
        for record in context.primary_benchmarks():
            for verdict in ordered_verdicts(record):
                total += 1
                succeeded = ran(verdict)
                eventually += succeeded
                first_try += succeeded and verdict.attempts <= 1

        pre = rate(first_try, total)
        post = rate(eventually, total)
        return {
            "pre_repair_success_rate": pre,
            "post_repair_success_rate": post,
            "repair_uplift": None if pre is None or post is None else post - pre,
            "suggestions": total,
        }
