"""verified / differs / unverified, as three separate rates.

Collapsing them would be the easy mistake. `differs` is a wrong answer that
often looks like a fast one; `unverified` is not an answer at all, and it is
never counted as correct here. Reporting the three side by side is what keeps
a preset that got fast by computing something else from reading as a winner.
"""
from __future__ import annotations

from jumper_ablations.metrics.base import (
    CATEGORY_SUGGESTIONS,
    METHOD_DETERMINISTIC,
    SCOPE_CELL,
    DeterministicMetric,
    MetricSpec,
)
from jumper_ablations.metrics.verdicts import (
    DIFFERS,
    UNVERIFIED,
    VERIFIED,
    ordered_verdicts,
    ran,
)
from jumper_ablations.statistics import rate


class CorrectnessRates(DeterministicMetric):
    spec = MetricSpec(
        id="correctness_rates",
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=(
            "verified_rate",
            "differs_rate",
            "unverified_rate",
            "failed_rate",
            "suggestions",
        ),
        description=(
            "Share of suggestions whose results matched the baseline, "
            "differed from it, could not be compared, or never ran. "
            "`unverified` is not counted as correct anywhere."
        ),
    )

    def compute(self, context, parameters: dict) -> dict:
        counts = {VERIFIED: 0, DIFFERS: 0, UNVERIFIED: 0}
        failed = 0
        total = 0
        for record in context.primary_benchmarks():
            for verdict in ordered_verdicts(record):
                total += 1
                if not ran(verdict):
                    failed += 1
                    continue
                counts[verdict.correctness] = counts.get(verdict.correctness, 0) + 1

        return {
            "verified_rate": rate(counts.get(VERIFIED, 0), total),
            "differs_rate": rate(counts.get(DIFFERS, 0), total),
            "unverified_rate": rate(counts.get(UNVERIFIED, 0), total),
            "failed_rate": rate(failed, total),
            "suggestions": total,
        }
