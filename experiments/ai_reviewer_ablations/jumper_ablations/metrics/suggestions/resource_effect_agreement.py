"""Did the resource a suggestion promised to relieve actually get relieved?

A hybrid on purpose, and the split is the point. The judge reads only the
suggestion's own description and states what it predicts - CPU down, GPU up,
memory unchanged. The harness then checks that prediction against the metrics
the benchmark measured. Nobody is asked to both predict and grade.

A suggestion whose speedup is real but whose stated mechanism did not happen
is a suggestion that worked by accident, and a user reading the description
learned something false.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from jumper_ablations.metrics.base import (
    CATEGORY_SUGGESTIONS,
    METHOD_JUDGE,
    SCOPE_RUN,
    JudgeMetric,
    MetricSpec,
)
from jumper_ablations.records.schema import BASELINE_LABEL
from jumper_ablations.statistics import rate

# Which measured columns stand for which resource. The benchmark summarises
# every collected metric as <name>_mean / <name>_max, so a prefix match is
# enough and survives a collector being added.
_RESOURCE_KEYS = {
    "cpu": ("cpu_util", "cpu_percent", "cpu"),
    "gpu": ("gpu_util", "gpu"),
    "memory": ("memory", "rss", "mem"),
}

# Below this relative change, a resource counts as unchanged rather than as
# moving in either direction. Sampled metrics wander by a few percent between
# two runs of the same code.
_NOISE_BAND = 0.10


class ResourcePrediction(BaseModel):
    suggestion_index: int
    resource: str
    # What the description promises: "down", "up" or "none".
    direction: str
    quote: str = ""


class ResourceEffectVerdict(BaseModel):
    predictions: list[ResourcePrediction] = Field(default_factory=list)
    notes: str = ""


def _measured(metrics: dict, resource: str) -> float | None:
    """The mean value of the first column matching *resource*."""
    for prefix in _RESOURCE_KEYS.get(resource, ()):
        for name, value in (metrics or {}).items():
            if name.startswith(prefix) and name.endswith("_mean"):
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
    return None


def _direction(baseline: float | None, variant: float | None) -> str | None:
    if baseline is None or variant is None or baseline == 0:
        return None
    change = (variant - baseline) / abs(baseline)
    if abs(change) < _NOISE_BAND:
        return "none"
    return "up" if change > 0 else "down"


class ResourceEffectAgreement(JudgeMetric):
    spec = MetricSpec(
        id="resource_effect_agreement",
        category=CATEGORY_SUGGESTIONS,
        evaluation_method=METHOD_JUDGE,
        scope=SCOPE_RUN,
        reported_values=(
            "agreement_rate",
            "checked_predictions",
            "uncheckable_predictions",
        ),
        description=(
            "Share of the resource changes a suggestion's description "
            "predicts that the measured metrics actually show, in the "
            "direction predicted. Predictions the benchmark did not measure "
            "are reported as uncheckable rather than counted either way."
        ),
    )
    verdict_model = ResourceEffectVerdict

    def from_verdict(self, verdict: ResourceEffectVerdict, context) -> dict:
        benchmarks = context.record.outputs.benchmarks
        baseline = benchmarks.get(BASELINE_LABEL)
        if baseline is None:
            return {
                "agreement_rate": None,
                "checked_predictions": 0,
                "uncheckable_predictions": len(verdict.predictions),
            }

        agreed = 0
        checked = 0
        uncheckable = 0
        for prediction in verdict.predictions:
            variant = benchmarks.get(str(prediction.suggestion_index))
            if variant is None or not variant.metrics:
                uncheckable += 1
                continue
            measured = _direction(
                _measured(baseline.metrics, prediction.resource),
                _measured(variant.metrics, prediction.resource),
            )
            if measured is None:
                uncheckable += 1
                continue
            checked += 1
            agreed += measured == prediction.direction

        return {
            "agreement_rate": rate(agreed, checked),
            "checked_predictions": checked,
            "uncheckable_predictions": uncheckable,
        }
