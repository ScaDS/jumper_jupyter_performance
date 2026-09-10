"""Do the replay modes agree about the same stored suggestions?

The fast replay modes exist to make a benchmark affordable, and this asks what
that costs in trust. The comparison is only meaningful because the modes score
one stored set of suggestions rather than each asking the model again: any
disagreement here is the instrument, not a new sample.

The fallback rate is reported for the same reason it is recorded at all - the
reviewer degrades a mode it cannot serve to the full replay with a warning
rather than failing, so a mode's results can silently be another mode's. Only
records whose *actual* mode is the requested one are compared.
"""
from __future__ import annotations

from itertools import combinations

from jumper_ablations.metrics.base import (
    CATEGORY_SUGGESTIONS,
    METHOD_DETERMINISTIC,
    SCOPE_CELL,
    DeterministicMetric,
    MetricSpec,
)
from jumper_ablations.metrics.verdicts import verdicts_by_index
from jumper_ablations.statistics import dispersion, mean, rate, spearman


class ReplayModeAgreement(DeterministicMetric):
    spec = MetricSpec(
        id="replay_mode_agreement",
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=(
            "modes_compared",
            "correctness_agreement",
            "mean_speedup_dispersion",
            "mean_rank_correlation",
            "fallback_rate",
        ),
        description=(
            "Across replay modes measuring the same stored suggestions: how "
            "often they assign the same correctness verdict, how far apart "
            "their speedups are, how similarly they rank, and how often a "
            "requested mode fell back to the full replay."
        ),
    )

    def compute(self, context, parameters: dict) -> dict:
        benchmarks = context.benchmarks()
        fallbacks = sum(1 for record in benchmarks if record.environment.degraded)
        fallback_rate = rate(fallbacks, len(benchmarks))

        modes = context.measured_modes()
        if len(modes) < 2:
            return {
                "modes_compared": len(modes),
                "fallback_rate": fallback_rate,
            }

        # One row per (generation, mode): the same suggestions, timed twice.
        per_generation: dict[int, dict[str, object]] = {}
        for record in benchmarks:
            mode = record.environment.actual_replay_mode
            if not mode:
                continue
            per_generation.setdefault(record.identity.generation, {})[mode] = record

        agreements = []
        dispersions = []
        correlations = []

        for by_mode in per_generation.values():
            for left_mode, right_mode in combinations(sorted(by_mode), 2):
                left = verdicts_by_index(by_mode[left_mode])
                right = verdicts_by_index(by_mode[right_mode])
                shared = sorted(set(left) & set(right))
                if not shared:
                    continue

                agreements.extend(
                    left[index].correctness == right[index].correctness
                    for index in shared
                )
                for index in shared:
                    spread = dispersion(
                        [left[index].speedup, right[index].speedup]
                    )
                    if spread is not None:
                        dispersions.append(spread)

                correlation = spearman(
                    [left[index].speedup for index in shared],
                    [right[index].speedup for index in shared],
                )
                if correlation is not None:
                    correlations.append(correlation)

        return {
            "modes_compared": len(modes),
            "correctness_agreement": mean([float(one) for one in agreements]),
            "mean_speedup_dispersion": mean(dispersions),
            "mean_rank_correlation": mean(correlations),
            "fallback_rate": fallback_rate,
        }
