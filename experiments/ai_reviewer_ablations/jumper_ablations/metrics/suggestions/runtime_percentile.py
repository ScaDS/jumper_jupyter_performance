"""ECCO's Runtime %: where the rewrite lands among every user's submission.

The share of the problem's CodeNet submissions that are slower than the
suggestion, ``|{s : T(s) > T(suggestion)}| / |submissions|``. As in ECCO the
spectrum is the submissions that run: one that crashes or prints a wrong
answer is not a point on it. A submission that hit the timeout is on it, as
slower than anything that finished. Averaged over correct suggestions.
"""

from __future__ import annotations

from jumper_ablations.metrics.base import (
    CATEGORY_SUGGESTIONS,
    METHOD_DETERMINISTIC,
    SCOPE_CELL,
    DeterministicMetric,
    MetricSpec,
)
from jumper_ablations.statistics import mean
from jumper_ablations.verification.schema import STATUS_TIMEOUT
from jumper_ablations.verification.views import (
    correct_suggestions,
    references_of,
)


def spectrum(population, field: str) -> list[float]:
    """The population's values of *field*, a timeout counted as infinite."""
    values = []
    for measurement in population:
        if measurement.status == STATUS_TIMEOUT:
            values.append(float("inf"))
        elif measurement.ok and measurement.correct:
            value = getattr(measurement, field)
            if value is not None:
                values.append(float(value))
    return values


def percentile(values: list[float], own: float | None) -> float | None:
    """Percentage of *values* strictly above *own*."""
    if own is None or not values:
        return None
    return 100.0 * sum(1 for value in values if value > own) / len(values)


class RuntimePercentile(DeterministicMetric):
    spec = MetricSpec(
        id="runtime_percentile",
        display_name="Runtime %",
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=(
            "mean_runtime_percentile",
            "correct_suggestions",
            "submissions",
        ),
        description=(
            "Percentage of CodeNet submissions slower than the suggestion, "
            "mean over correct suggestions (ECCO)."
        ),
        supporting_values=("correct_suggestions", "submissions"),
    )

    def compute(self, context, parameters: dict) -> dict:
        references = references_of(context, "ecco")
        if references is None:
            return {
                "mean_runtime_percentile": None,
                "correct_suggestions": 0,
                "submissions": 0,
            }
        values = spectrum(references.population, "time_s")
        correct = correct_suggestions(context)
        return {
            "mean_runtime_percentile": mean(
                percentile(values, measurement.time_s) for measurement in correct
            ),
            "correct_suggestions": len(correct),
            "submissions": len(values),
        }
