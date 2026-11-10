"""verified / differs / unverified, as three separate rates.

Collapsing them would be the easy mistake. `differs` is a wrong answer that
often looks like a fast one; `unverified` is not an answer at all, and it is
never counted as correct here. Reporting the three side by side is what keeps
a preset that got fast by computing something else from reading as a winner.

This is the **reviewer's own** verdict, deliberately: it is a diagnostic of
the product, and the point of it is the `unverified` column, which says how
often the reviewer could not tell its user whether a rewrite was safe. It is
not the experiment's correctness. Where `cli.verify` ran, that is, and
`correctness_conditioned_speedup.coverage` reports it - on the ECCO pilot the
two disagreed about 91 of 268 suggestions in both directions.
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
        display_name="Correctness Status Rates",
        value_names={
            "verified_rate": "Correct output",
            "differs_rate": "Wrong output",
            "unverified_rate": "Not comparable",
            "failed_rate": "Did not run",
        },
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
            "differed from it, could not be compared, or never ran, as the "
            "reviewer itself judged them. `unverified` is not counted as "
            "correct anywhere. For the experiment's correctness read "
            "`correctness_conditioned_speedup.coverage`, which follows "
            "cli.verify."
        ),
        supporting_values=(
            "suggestions",
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
                counts[verdict.correctness] = (
                    counts.get(verdict.correctness, 0) + 1
                )

        return {
            "verified_rate": rate(counts.get(VERIFIED, 0), total),
            "differs_rate": rate(counts.get(DIFFERS, 0), total),
            "unverified_rate": rate(counts.get(UNVERIFIED, 0), total),
            "failed_rate": rate(failed, total),
            "suggestions": total,
        }
