"""ECCO's Runtime %: where the rewrite lands among every user's submission.

The share of the problem's CodeNet submissions that are slower than the
suggestion, ``|{s : T(s) > T(suggestion)}| / |submissions|``. As in ECCO the
spectrum is the submissions that run: one that crashes or prints a wrong
answer is not a point on it. A submission that hit the timeout is on it, as
slower than anything that finished.

Both numbers are about the option a user would take, because that is what a
user takes: one suggestion out of one response. So each generation is reduced
to its best correct suggestion first, and the two values differ only in what
is done with those.

``mean_runtime_percentile`` - **Mean Runtime %** - the mean of them: what a
generation of this preset achieves, typically.

``best_runtime_percentile`` - **Best Runtime %** - the largest of them, not
averaged with anything: the best this preset was ever seen to do. It is an
extremum over the whole cell, so it rises with the number of generations and
can never be compared between presets drawn a different number of times.

Both are read next to ``correct_suggestions``: a percentile over the two
suggestions that happened to be correct is not a ranking of the preset.
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
    verified_reviews,
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


def best_per_review(context, values: list[float], field: str) -> list[float]:
    """Each generation's best correct percentile - the grain both percentiles use.

    *field* is the measured quantity ranked against *values*: ``time_s`` for
    Runtime %, ``peak_memory_bytes`` for Memory %.

    A generation with no correct suggestion is left out rather than counted as
    zero: a response that produced nothing usable is already reported by
    `coverage` and by `correct_suggestions`, and folding it in here as "0% of
    humans beaten" would mix two different failures into one number.
    """
    best = []
    for review in verified_reviews(context):
        ranked = [
            percentile(values, getattr(measurement, field))
            for measurement in review
            if measurement.ok and measurement.correct
        ]
        ranked = [value for value in ranked if value is not None]
        if ranked:
            best.append(max(ranked))
    return best


class RuntimePercentile(DeterministicMetric):
    spec = MetricSpec(
        id="runtime_percentile",
        display_name="Runtime %",
        value_names={
            "mean_runtime_percentile": "Mean Runtime %",
            "best_runtime_percentile": "Best Runtime %",
        },
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_DETERMINISTIC,
        scope=SCOPE_CELL,
        reported_values=(
            "mean_runtime_percentile",
            "best_runtime_percentile",
            "correct_suggestions",
            "submissions",
        ),
        description=(
            "Percentage of CodeNet submissions slower than the suggestion "
            "(ECCO), over each generation's best correct suggestion: their "
            "mean, and the single largest of them."
        ),
        supporting_values=("correct_suggestions", "submissions"),
    )

    def compute(self, context, parameters: dict) -> dict:
        references = references_of(context, "ecco")
        if references is None:
            return {
                "mean_runtime_percentile": None,
                "best_runtime_percentile": None,
                "correct_suggestions": 0,
                "submissions": 0,
            }
        values = spectrum(references.population, "time_s")
        best = best_per_review(context, values, "time_s")
        return {
            "mean_runtime_percentile": mean(best),
            "best_runtime_percentile": max(best) if best else None,
            "correct_suggestions": len(correct_suggestions(context)),
            "submissions": len(values),
        }
