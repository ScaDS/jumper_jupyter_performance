"""KernelBench's fast_p: correct and more than p times faster than eager.

``fast_p = |{s : correct(s) and T(eager) / T(s) > p}| / |suggestions|``, with
correctness by allclose and both times per call, CUDA-event timed, as
cli.verify measures them. ``fast_0`` is the correctness rate; ``fast_1`` is
correct and faster than PyTorch eager at all. Every suggestion is in the
denominator, including the ones that failed.
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
from jumper_ablations.verification.schema import EAGER
from jumper_ablations.verification.views import (
    ratio,
    references_of,
    verified_suggestions,
)

# The thresholds reported, as the names they are reported under.
THRESHOLDS = {
    "fast_0": 0.0,
    "fast_1": 1.0,
    "fast_1_5": 1.5,
    "fast_2": 2.0,
}


class FastP(DeterministicMetric):
    spec = MetricSpec(
        id="fast_p",
        display_name="fast_p",
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=(*THRESHOLDS, "suggestions"),
        description=(
            "Share of suggestions that are allclose-correct and more than p "
            "times faster than the eager Model per call (KernelBench)."
        ),
        supporting_values=("suggestions",),
    )

    def compute(self, context, parameters: dict) -> dict:
        references = references_of(context, "kernelbench")
        if references is None:
            return {**{name: None for name in THRESHOLDS}, "suggestions": 0}
        eager = references.entries.get(EAGER)
        baseline = eager.time_s if eager and eager.ok else None
        measured = verified_suggestions(context)
        speedups = [
            ratio(baseline, measurement.time_s)
            if measurement.ok and measurement.correct
            else None
            for measurement in measured
        ]
        values = {
            name: rate(
                sum(
                    1
                    for speedup in speedups
                    if speedup is not None and speedup > threshold
                ),
                len(measured),
            )
            for name, threshold in THRESHOLDS.items()
        }
        values["suggestions"] = len(measured)
        return values
