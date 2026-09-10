"""What the repair loop had to do, and whether it changed the answer.

The repair loop is generous: up to three rewrites per suggestion, triggered by
a syntax error, a crash, a timeout, or results that diverged. It makes a
preset look better than the model was, so how hard it had to work is part of
the preset's score.

The last number needs its definition stated. "Idea changed" is measured as the
suggestion's code differing after repair once formatting is normalised away,
comparing the review record against the benchmark record of the same
generation. That is a lower bound on a changed idea - a rename counts too -
and it is only available because the review and the benchmark are separate
invocations, so the suggestions are recorded as the model wrote them before
the repair loop overwrites them.
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
    normalised_code,
    ordered_verdicts,
    ran,
    suggestion_codes,
)
from jumper_ablations.statistics import mean, rate


class Repairs(DeterministicMetric):
    spec = MetricSpec(
        id="repairs",
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=(
            "attempts_per_suggestion",
            "repaired_share",
            "repair_success_rate",
            "idea_change_rate",
        ),
        description=(
            "Repair rounds spent per suggestion, the share of suggestions that "
            "needed one, how often repairing produced something that ran, and "
            "how often the repaired code differs from what the model wrote."
        ),
    )

    def compute(self, context, parameters: dict) -> dict:
        attempts = []
        repaired = 0
        repaired_ok = 0
        total = 0
        changed = 0
        compared = 0

        originals = {
            record.identity.generation: suggestion_codes(record)
            for record in context.reviews()
        }

        for record in context.primary_benchmarks():
            before = originals.get(record.identity.generation, {})
            after = suggestion_codes(record)
            for index, verdict in enumerate(ordered_verdicts(record), 1):
                total += 1
                attempts.append(max(verdict.attempts - 1, 0))
                if verdict.attempts > 1:
                    repaired += 1
                    repaired_ok += ran(verdict)
                    if index in before and index in after:
                        compared += 1
                        changed += normalised_code(before[index]) != normalised_code(
                            after[index]
                        )

        return {
            "attempts_per_suggestion": mean(attempts),
            "repaired_share": rate(repaired, total),
            "repair_success_rate": rate(repaired_ok, repaired),
            "idea_change_rate": rate(changed, compared),
        }
