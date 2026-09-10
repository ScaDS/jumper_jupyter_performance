"""How many suggestions were asked for, and how many came back.

A preset that answers with one option when three were requested has not just
been less useful - it has also made every per-suggestion rate below rest on a
smaller sample, so this is read before them, not after.
"""
from __future__ import annotations

from jumper_ablations.metrics.base import (
    CATEGORY_SUGGESTIONS,
    METHOD_DETERMINISTIC,
    SCOPE_CELL,
    DeterministicMetric,
    MetricSpec,
)
from jumper_ablations.statistics import mean, rate

DEFAULT_REQUESTED = 3


class RequestedVersusReturned(DeterministicMetric):
    spec = MetricSpec(
        id="requested_vs_returned",
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=(
            "requested",
            "mean_returned",
            "mean_shortfall",
            "compliance_rate",
        ),
        description=(
            "Suggestions requested versus returned per generation, and the "
            "fraction of generations that returned at least the requested "
            "number. `requested` comes from the metric's config: nothing in "
            "the pipeline pins it."
        ),
    )

    def compute(self, context, parameters: dict) -> dict:
        requested = int(parameters.get("requested", DEFAULT_REQUESTED))
        counts = [
            len(record.outputs.suggestions) for record in context.reviews()
        ]
        if not counts:
            return {"requested": requested}

        compliant = sum(1 for count in counts if count >= requested)
        return {
            "requested": requested,
            "mean_returned": mean(counts),
            "mean_shortfall": mean([max(requested - count, 0) for count in counts]),
            "compliance_rate": rate(compliant, len(counts)),
        }
