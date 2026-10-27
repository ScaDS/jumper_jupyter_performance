"""KernelBench's speedup over torch.compile, beside the one over eager.

``T(torch.compile Model) / T(suggestion)`` and ``T(eager Model) /
T(suggestion)``, per call as cli.verify times them, geometric mean over
allclose-correct suggestions. The first asks whether the rewrite beat the
one-line optimisation PyTorch already offers; the second is KernelBench's
headline speedup, timed per call rather than over the whole cell.
"""

from __future__ import annotations

from jumper_ablations.metrics.base import (
    CATEGORY_SUGGESTIONS,
    METHOD_DETERMINISTIC,
    SCOPE_CELL,
    DeterministicMetric,
    MetricSpec,
)
from jumper_ablations.statistics import geometric_mean
from jumper_ablations.verification.schema import EAGER, TORCH_COMPILE
from jumper_ablations.verification.views import (
    correct_suggestions,
    ratio,
    references_of,
)


class CompileSpeedup(DeterministicMetric):
    spec = MetricSpec(
        id="compile_speedup",
        display_name="Speedup vs torch.compile",
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=(
            "geometric_mean_compile_speedup",
            "geometric_mean_eager_speedup",
            "correct_suggestions",
        ),
        description=(
            "T(torch.compile) / T(suggestion) and T(eager) / T(suggestion), "
            "per call, geometric mean over allclose-correct suggestions "
            "(KernelBench)."
        ),
        supporting_values=("correct_suggestions",),
        ratio_values=(
            "geometric_mean_compile_speedup",
            "geometric_mean_eager_speedup",
        ),
    )

    def compute(self, context, parameters: dict) -> dict:
        references = references_of(context, "kernelbench")
        if references is None:
            return {
                "geometric_mean_compile_speedup": None,
                "geometric_mean_eager_speedup": None,
                "correct_suggestions": 0,
            }

        def baseline(label: str) -> float | None:
            measurement = references.entries.get(label)
            return measurement.time_s if measurement and measurement.ok else None

        correct = correct_suggestions(context)
        return {
            "geometric_mean_compile_speedup": geometric_mean(
                ratio(baseline(TORCH_COMPILE), measurement.time_s)
                for measurement in correct
            ),
            "geometric_mean_eager_speedup": geometric_mean(
                ratio(baseline(EAGER), measurement.time_s)
                for measurement in correct
            ),
            "correct_suggestions": len(correct),
        }
