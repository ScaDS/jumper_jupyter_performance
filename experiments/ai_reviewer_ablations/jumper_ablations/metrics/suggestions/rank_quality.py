"""Was the model's ordering of its own options any good?

The suggest prompt asks for options ordered from most to least impactful, and
a user reads them in that order. If the ordering carries no information, the
first option is just a random one and `pass@1` is luck.

Two views: how often the option ranked first was the fastest correct one, and
how well the whole proposed order correlates with the measured one.
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
from jumper_ablations.statistics import mean, rate, spearman


class RankQuality(DeterministicMetric):
    spec = MetricSpec(
        id="rank_quality",
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=(
            "first_is_fastest_rate",
            "mean_rank_correlation",
            "comparable_generations",
        ),
        description=(
            "How often the first-ranked suggestion was the fastest correct "
            "one, and the mean Spearman correlation between the proposed "
            "order and the measured speedups. Generations with fewer than two "
            "correct suggestions cannot be ranked and are excluded from the "
            "correlation."
        ),
    )

    def compute(self, context, parameters: dict) -> dict:
        first_is_fastest = 0
        ranked_generations = 0
        correlations = []

        for record in context.primary_benchmarks():
            verdicts = ordered_verdicts(record)
            correct = [
                (position, verdict.speedup)
                for position, verdict in enumerate(verdicts)
                if is_correct(verdict) and verdict.speedup is not None
            ]
            if not correct:
                continue
            ranked_generations += 1
            fastest_position = max(correct, key=lambda pair: pair[1])[0]
            first_is_fastest += fastest_position == 0

            if len(correct) >= 2:
                positions = [position for position, _ in correct]
                # Negated: rank 0 is the model's best guess, so agreement means
                # earlier positions carry larger speedups.
                speedups = [-value for _, value in correct]
                correlation = spearman(positions, speedups)
                if correlation is not None:
                    correlations.append(correlation)

        return {
            "first_is_fastest_rate": rate(first_is_fastest, ranked_generations),
            "mean_rank_correlation": mean(correlations),
            "comparable_generations": len(correlations),
        }
